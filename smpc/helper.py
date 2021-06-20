import abc
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np
import rsa

from federated_pure_regression_survival_svm.model import ObjectivesW

MAX_RAND_INT: int = 234_234_234_324

@dataclass
class SMPCMasked:
    masked_data: Any
    encrypted_masks: Dict[int, bytes]


class SMPCMaskingHelper(abc.ABC):
    def __init__(self, data: Any):
        self.data = data

    @staticmethod
    def encrypt_mask(mask: np.array, public_key: rsa.PublicKey) -> bytes:
        mask = mask.astype(np.float64).tobytes()
        return rsa.encrypt(mask, public_key)

    @abc.abstractmethod
    def mask(self, pub_keys_of_other_parties: Dict[int, rsa.PublicKey]) -> SMPCMasked:
        pass


class MaskingHelperNpArray(SMPCMaskingHelper):
    def __init__(self, data: np.array):
        super().__init__(data)

    def mask(self, pub_keys_of_other_parties: Dict[int, rsa.PublicKey]):
        encrypted_masks: Dict[int, bytes] = dict.fromkeys(pub_keys_of_other_parties.keys())
        masked_data = self.data.copy()
        noise_guard = self.data.copy()
        noise_guard[noise_guard > 1] = 1
        NOISE_MAX = noise_guard * MAX_RAND_INT
        for client_id, client_pub_key in pub_keys_of_other_parties.items():
            mask: np.array = np.random.randint(NOISE_MAX, size=self.data.shape)
            print(mask)
            masked_data -= mask
            encrypted_masks[client_id] = self.encrypt_mask(mask, client_pub_key)
        return SMPCMasked(masked_data=masked_data, encrypted_masks=encrypted_masks)


class MaskedObjectivesW(SMPCMaskingHelper):
    def __init__(self, data: ObjectivesW):
        super().__init__(data)

    def mask(self, pub_keys_of_other_parties: Dict[int, rsa.PublicKey]) -> SMPCMasked:
        self.data: ObjectivesW
        transformed = np.concatenate([[self.data.local_sum_of_zeta_squared], self.data.local_gradient])
        return MaskingHelperNpArray(transformed).mask(pub_keys_of_other_parties)


class SMPCClient(object):
    def __init__(self, client_id):
        self.pub_key: rsa.PublicKey
        self._priv_key: rsa.PrivateKey
        self.pub_key, self._priv_key = rsa.newkeys(1024)

        self.client_id = client_id
        self.pub_keys_of_other_parties: Dict[int, rsa.PublicKey] = {self.client_id: self.pub_key}

    def decrypt_mask(self, encrypted_mask: bytes) -> np.array:
        return np.frombuffer(rsa.decrypt(encrypted_mask, self._priv_key), dtype=np.float64)

    def add_party_pub_key(self, party_id: int, pub_key: rsa.PublicKey):
        self.pub_keys_of_other_parties[party_id] = pub_key

    def add_party_pub_key_dict(self, update: Dict[int, rsa.PublicKey]):
        self.pub_keys_of_other_parties.update(update)

    def get(self) -> Tuple[int, rsa.PublicKey]:
        return self.client_id, self.pub_key

    def mask(self, masking_helper: SMPCMaskingHelper):
        return masking_helper.mask(self.pub_keys_of_other_parties)

    def sum_encrypted_masks_up(self, encrypted_masks: List[bytes]) -> int:
        summed_masks: int = 0
        for encrypted_mask in encrypted_masks:
            decrypted_mask = self.decrypt_mask(encrypted_mask)
            summed_masks += decrypted_mask

        return summed_masks


if __name__ == '__main__':
    data = ObjectivesW(
        local_gradient=np.array([4.03434, 0.002323232, 0.00000000012]),
        local_sum_of_zeta_squared=1000
    )
    clients = []
    pub_keys = {}
    for i in range(3):
        client = SMPCClient(i)
        clients.append(client)
        pub_keys[i] = client.pub_key

    print(pub_keys)
    for client in clients:
        client.add_party_pub_key_dict(pub_keys)

    # expected = np.zeros_like(data)
    res: List[SMPCMasked] = []
    for client in clients:
        local_data = data
        res.append(client.mask(MaskedObjectivesW(local_data)))
        # expected += local_data

    # print(expected)
    print(res)

    smpc_masked: SMPCMasked
    aggregated_masked_result = np.zeros_like(res[0].masked_data)
    for smpc_masked in res:
        print(smpc_masked.masked_data)
        aggregated_masked_result += smpc_masked.masked_data


    def distribute_masks(masked_results: List[SMPCMasked]) -> Dict[int, List[bytes]]:
        shares = defaultdict(list)
        for masked in masked_results:
            enc_masks: Dict[int, bytes] = masked.encrypted_masks
            for recipient, enc_mask in enc_masks.items():
                shares[recipient].append(enc_mask)
        return dict(shares)

    shares = distribute_masks(res)
    print(shares)

    local_mask_sum = []
    for client in clients:
        local_mask_sum.append(client.sum_encrypted_masks_up(shares[client.client_id]))

    print(local_mask_sum)

    shape = aggregated_masked_result.shape
    aggregated_mask = np.zeros_like(res[0].masked_data, dtype=np.float64)
    mask: np.array
    for mask in local_mask_sum:
        aggregated_mask += mask.reshape(shape)

    print(aggregated_masked_result)
    print(aggregated_mask)

    print((aggregated_masked_result + aggregated_mask)/3)
