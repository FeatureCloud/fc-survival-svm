import shutil
import threading
import time

import joblib
import jsonpickle
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from app.algo import Coordinator, Client


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.client = None
        self.input = None
        self.sep = None
        self.label_column = None
        self.test_size = None
        self.random_state = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.beta = None
        self.iter_counter = 0
        self.beta_finished = None
        self.max_iter = None

    def read_config(self):
        with open(self.INPUT_DIR + '/config.yml') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)['fc_logistic_regression']
            self.input = config['files']['input']
            self.sep = config['files']['sep']
            self.label_column = config['files']['label_column']
            self.test_size = config['evaluation']['test_size']
            self.random_state = config['evaluation']['random_state']
            self.max_iter = 10000

        shutil.copyfile(self.INPUT_DIR + '/config.yml', self.OUTPUT_DIR + '/config.yml')

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....")
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read_input = 2
        state_preprocessing = 3
        state_local_computation = 4
        state_wait_for_aggregation = 5
        state_global_aggregation = 6
        state_writing_results = 7
        state_finishing = 8

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:
            if state == state_initializing:
                print("Initializing")
                if self.id is not None:  # Test if setup has happened already
                    print("Coordinator", self.coordinator)
                    if self.coordinator:
                        self.client = Coordinator()
                    else:
                        self.client = Client()
                    state = state_read_input

            if state == state_read_input:
                print('Read input and config')
                self.read_config()

                numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
                self.X = pd.read_csv(self.INPUT_DIR + "/" + self.input, sep=self.sep).select_dtypes(
                    include=numerics).dropna()
                self.y = self.X.loc[:, self.label_column]
                self.X = self.X.drop(self.label_column, axis=1)

                if self.test_size is not None:
                    self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y,
                                                                                test_size=self.test_size,
                                                                                random_state=self.random_state)
                state = state_preprocessing

            if state == state_preprocessing:
                self.X, self.y, self.beta = self.client.init(self.X, self.y)
                state = state_local_computation

            if state == state_local_computation:
                print("Perform local beta update")
                self.progress = 'local beta update'
                self.iter_counter = self.iter_counter + 1
                try:
                    data_to_send = self.client.compute_derivatives(self.X, self.y, self.beta)
                except FloatingPointError:
                    data_to_send = "early_stop"

                data_to_send = jsonpickle.encode(data_to_send)

                if self.coordinator:
                    self.data_incoming.append(data_to_send)
                    state = state_global_aggregation
                else:
                    self.data_outgoing = data_to_send
                    self.status_available = True
                    state = state_wait_for_aggregation
                    print(f'[CLIENT] Sending computation data to coordinator', flush=True)

            if state == state_wait_for_aggregation:
                print("Wait for aggregation")
                self.progress = 'wait for aggregation'
                if len(self.data_incoming) > 0:
                    print("Received aggregation data from coordinator.")
                    data_decoded = jsonpickle.decode(self.data_incoming[0])
                    self.beta, self.beta_finished = data_decoded[0], data_decoded[1]
                    self.data_incoming = []
                    if not self.beta_finished and self.iter_counter < self.max_iter:
                        state = state_local_computation
                    else:
                        print("Beta update finished.")
                        self.client.set_coefs(self.beta)
                        state = state_writing_results

            # GLOBAL PART

            if state == state_global_aggregation:
                print("Global computation")
                self.progress = 'computing...'
                if len(self.data_incoming) == len(self.clients):
                    data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
                    self.data_incoming = []
                    self.beta, self.beta_finished = self.client.aggregate_beta(data)
                    data_to_broadcast = jsonpickle.encode([self.beta, self.beta_finished])
                    self.data_outgoing = data_to_broadcast
                    self.status_available = True
                    print(f'[COORDINATOR] Broadcasting computation data to clients', flush=True)
                    if not self.beta_finished and self.iter_counter < self.max_iter:
                        state = state_local_computation
                    else:
                        print("Beta update finished.")
                        self.client.set_coefs(self.beta)
                        state = state_writing_results

            if state == state_writing_results:
                print("Writing results")
                # now you can save it to a file
                print("Coef:", self.client.coef_)
                joblib.dump(self.client, self.OUTPUT_DIR + '/model.pkl')
                model = self.client

                if self.test_size is not None:
                    y_pred = model.predict(self.X_test)
                    y_proba = model.predict_proba(self.X_test)
                    self.y_test.to_csv(self.OUTPUT_DIR + "/y_test.csv", index=False)
                    pd.DataFrame(y_proba).to_csv(self.OUTPUT_DIR + "/y_proba.csv", index=False)
                    pd.DataFrame(y_pred).to_csv(self.OUTPUT_DIR + "/y_pred.csv", index=False)

                state = state_finishing

            if state == state_finishing:
                print("Finishing")
                self.progress = 'finishing...'
                if self.coordinator:
                    time.sleep(10)
                self.status_finished = True
                break

            time.sleep(1)


logic = AppLogic()
