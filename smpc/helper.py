import abc
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional, Type

import numpy as np
import rsa
from rsa import PrivateKey

from federated_pure_regression_survival_svm.model import ObjectivesW, DataDescription, ObjectivesS

MAX_RAND_INT: int = 234_234_234_324
D_TYPE: Type = np.int


class SMPCMasked(abc.ABC):
    def __init__(self):
        self.attributes: Dict = {}
        self.inner_representation: Any = None
        self.encrypted_masks: Dict[int, List[SMPCEncryptedMask]] = defaultdict(list)

    @abc.abstractmethod
    def unmasked_obj(self, summed_up_masks):
        pass

    @abc.abstractmethod
    def mask(self, data: Any, pub_keys_of_other_parties: Dict[int, rsa.PublicKey]):
        pass


class MaskedDataDescription(SMPCMasked):
    def unmasked_obj(self, summed_up_masks):
        decrypted_inner = self.inner_representation + summed_up_masks
        return DataDescription(n_samples=decrypted_inner[0],
                               n_features=self.attributes['n_features'],
                               sum_of_times=decrypted_inner[1])

    def mask(self, data: DataDescription, pub_keys_of_other_parties: Dict[int, rsa.PublicKey]):
        self.inner_representation = np.array([data.n_samples, data.sum_of_times])
        for client_id, client_pub_key in pub_keys_of_other_parties.items():
            mask = SMPCMask(self.inner_representation.shape)
            logging.debug(mask)
            self.inner_representation = mask.apply(self.inner_representation)
            self.encrypted_masks[client_id].append(mask.encrypt(client_pub_key))
        self.attributes['n_features'] = data.n_features
        return self

    def __add__(self, other):
        assert isinstance(other, self.__class__)
        assert self.attributes['n_features'] == other.attributes['n_features']
        self.inner_representation += other.inner_representation
        for client_id in self.encrypted_masks:
            self.encrypted_masks[client_id].append(other.encrypted_masks[client_id][0])
        return self

    def __repr__(self):
        return f"{self.encrypted_masks}, {self.inner_representation}"


class MaskedObjectivesW(SMPCMasked):
    def unmasked_obj(self, summed_up_masks):
        decrypted_inner = self.inner_representation + summed_up_masks
        return ObjectivesW(local_sum_of_zeta_squared=decrypted_inner[0],
                           local_gradient_update=decrypted_inner[1:])

    def mask(self, data: ObjectivesW, pub_keys_of_other_parties: Dict[int, rsa.PublicKey]):
        self.inner_representation = np.hstack([data.local_sum_of_zeta_squared, data.local_gradient_update])
        for client_id, client_pub_key in pub_keys_of_other_parties.items():
            mask = SMPCMask(self.inner_representation.shape)
            logging.debug(mask)
            self.inner_representation = mask.apply(self.inner_representation)
            logging.debug(f"client_id: {client_id}")
            logging.debug(f"client pub_key: {client_pub_key}")
            encrypted_mask = mask.encrypt(client_pub_key)
            logging.debug(f"encrypted mask: {encrypted_mask}")
            self.encrypted_masks[client_id].append(mask.encrypt(client_pub_key))
            logging.debug(f"self.encrypted_masks: {self.encrypted_masks}")
        return self

    def __add__(self, other):
        assert isinstance(other, self.__class__)
        self.inner_representation += other.inner_representation
        for client_id in self.encrypted_masks:
            self.encrypted_masks[client_id].append(other.encrypted_masks[client_id][0])
        return self

    def __repr__(self):
        return f"{self.encrypted_masks}, {self.inner_representation}"


class MaskedObjectivesS(SMPCMasked):
    def unmasked_obj(self, summed_up_masks):
        decrypted_inner = self.inner_representation + summed_up_masks
        return ObjectivesS(local_hessp_update=decrypted_inner)

    def mask(self, data: ObjectivesS, pub_keys_of_other_parties: Dict[int, rsa.PublicKey]):
        self.inner_representation = np.array(data.local_hessp_update)
        for client_id, client_pub_key in pub_keys_of_other_parties.items():
            mask = SMPCMask(self.inner_representation.shape)
            mask.mask = mask.mask
            logging.debug(mask)
            self.inner_representation = mask.apply(self.inner_representation)
            self.encrypted_masks[client_id].append(mask.encrypt(client_pub_key))
        return self

    def __add__(self, other):
        assert isinstance(other, self.__class__)
        self.inner_representation += other.inner_representation
        for client_id in self.encrypted_masks:
            self.encrypted_masks[client_id].append(other.encrypted_masks[client_id][0])
        return self

    def __repr__(self):
        return f"{self.encrypted_masks}, {self.inner_representation}"


class SMPCEncryptedMask(object):
    def __init__(self, encrypted_mask: bytes):
        self.encrypted_mask = encrypted_mask

    def decrypt(self, private_key: PrivateKey):
        return np.frombuffer(rsa.decrypt(self.encrypted_mask, private_key), dtype=D_TYPE)


class SMPCMask(object):
    def __init__(self, shape_of_data_to_mask):
        self.mask = np.random.randint(MAX_RAND_INT, size=shape_of_data_to_mask)

    def apply(self, data):
        return data - self.mask.astype(dtype=np.float)

    def encrypt(self, public_key: rsa.PublicKey) -> SMPCEncryptedMask:
        mask = self.mask.astype(dtype=D_TYPE).tobytes()
        logging.debug(f"mask as bytes{mask}")
        return SMPCEncryptedMask(rsa.encrypt(mask, public_key))

    def __str__(self):
        return f"SMPCMask<{self.mask.tolist()}>"

    def __repr__(self):
        return self.__str__()


@dataclass
class SMPCRequest(object):
    data: Dict[str, Dict[int, List[Optional[SMPCEncryptedMask]]]]


class SmpcKeyManager(object):
    def __init__(self, client_id, key_size=4096, random_seed=None):
        np.random.seed(random_seed)

        self._pub_key: rsa.PublicKey
        self._priv_key: rsa.PrivateKey
        logging.info("Generating key pair")
        self._pub_key, self._priv_key = rsa.newkeys(key_size)
        logging.info("Finished generating key pair")

        self.client_id = client_id
        self.public_keys: Dict[str, rsa.PublicKey] = {self.client_id: self._pub_key}

    def add_party_pub_key(self, party_id: str, pub_key: rsa.PublicKey):
        """Add the public key of another party."""
        self.public_keys[party_id] = pub_key

    def add_party_pub_key_dict(self, update: Dict[str, rsa.PublicKey]):
        """Add public key of another party using a dict."""
        self.public_keys.update(update)

    def get_pubkey(self) -> Tuple[str, rsa.PublicKey]:
        """Return own public key."""
        return self.client_id, self.public_keys[self.client_id]

    def sum_encrypted_masks_up(self, encrypted_masks: List[SMPCEncryptedMask]) -> int:
        summed_masks: int = 0
        for encrypted_mask in encrypted_masks:
            decrypted_mask = encrypted_mask.decrypt(self._priv_key)
            summed_masks += decrypted_mask

        return summed_masks


if __name__ == '__main__':
    clients = []
    pub_keys = {}
    for i in range(3):
        client = SmpcKeyManager(i)
        clients.append(client)
        pub_keys[i] = client._pub_key

    print(pub_keys)
    for client in clients:
        client.add_party_pub_key_dict(pub_keys)

    masks = []
    for client in clients:
        dd = DataDescription(n_features=10, n_samples=100, sum_of_times=1000)
        masked_dd = MaskedDataDescription().mask(dd, client.public_keys)
        print(masked_dd)
        print(masked_dd.unmasked_obj([0, 0]))

    print()
