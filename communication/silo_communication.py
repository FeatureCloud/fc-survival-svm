import abc
import logging
import threading
import time

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy

jsonpickle_numpy.register_handlers()


class Communication(abc.ABC):
    def __init__(self):
        self.is_smpc: bool = False

        self.is_coordinator = None
        self.num_clients = None

        self.status_available = False  # Indicates whether there is data to share, if True make sure self.data_out is available
        self.data_incoming = []
        self._incoming_lock = threading.Lock()
        self.data_outgoing = None

    def init(self, is_coordinator: bool, num_clients: int):
        self.is_coordinator = is_coordinator
        self.num_clients = num_clients

    def clear(self):
        self.data_outgoing = None
        self.status_available = False

    def handle_incoming(self, data):
        # This method is called when new data arrives
        logging.info("Process incoming data....")
        self._incoming_lock.acquire()
        logging.debug("Adding data to data_incoming")
        self.data_incoming.append(data.read())
        logging.debug(f"data_incoming length: {len(self.data_incoming)}")
        logging.debug(f"data_incoming: {self.data_incoming}")
        self._incoming_lock.release()

    def handle_outgoing(self):
        logging.info("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def _assert_initialized(self):
        if self.is_coordinator is None or self.num_clients is None:
            raise Exception("Communication object is not properly initialized.")

    def _assert_is_coordinator(self):
        if not self.is_coordinator:
            raise Exception("This node is not the coordinator. An unexpected function was called.")

    def _send_local_channel(self, data):
        self.data_incoming.append(data)

    def _broadcast(self, data):
        self.data_outgoing = data
        self.status_available = True

    @abc.abstractmethod
    def send_to_coordinator(self, data):
        pass

    @abc.abstractmethod
    def broadcast(self, data):
        pass

    @abc.abstractmethod
    def wait_for_data_from_all(self, timeout: int = 3):
        pass

    @abc.abstractmethod
    def wait_for_data(self, timeout: int = 3):
        pass


class JsonEncodedCommunication(Communication):
    def _encode(self, data):
        return jsonpickle.encode(data)

    def _decode(self, encoded_data):
        return jsonpickle.decode(encoded_data)

    def send_to_coordinator(self, data):
        """Data is send to communicator"""
        self._assert_initialized()
        encoded_data = self._encode(data)

        if self.is_coordinator:
            self._send_local_channel(encoded_data)
        else:
            self._broadcast(encoded_data)

    def broadcast(self, data):
        """Data is send to each node"""
        self._assert_initialized()
        encoded_data = self._encode(data)

        if self.is_coordinator:
            self._send_local_channel(encoded_data)

        self._broadcast(encoded_data)

    def wait_for_data_from_all(self, timeout: int = 3):
        self._assert_initialized()
        self._assert_is_coordinator()

        while True:
            self._incoming_lock.acquire()
            if len(self.data_incoming) == self.num_clients:
                logging.debug("Received response of all nodes")

                decoded_data = [self._decode(client_data) for client_data in self.data_incoming]
                self.data_incoming = []

                self._incoming_lock.release()
                return decoded_data
            elif len(self.data_incoming) > self.num_clients:
                logging.error("Received more data than expected")
                raise Exception("Received more data than expected")
            else:
                logging.debug(f"Waiting for all nodes to respond. {len(self.data_incoming)} of {self.num_clients}...")

            self._incoming_lock.release()
            time.sleep(timeout)

    def wait_for_data(self, timeout: int = 3):
        self._assert_initialized()

        while True:
            self._incoming_lock.acquire()
            if len(self.data_incoming) == 1:
                logging.debug("Received response")
                data = self.data_incoming[0]
                logging.debug(data)
                decoded_data = self._decode(data)
                self.data_incoming = []

                self._incoming_lock.release()
                return decoded_data
            elif len(self.data_incoming) > 1:
                logging.error("Received more data than expected")
                raise Exception("Received more data than expected")
            else:
                logging.debug("Waiting for node response")

            self._incoming_lock.release()
            time.sleep(timeout)


class SmpcCommunication(JsonEncodedCommunication):
    def __init__(self):
        super().__init__()
        self.is_smpc = True
