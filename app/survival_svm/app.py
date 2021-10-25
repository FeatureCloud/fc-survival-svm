import json
from dataclasses import dataclass

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
from nptyping import NDArray, Float64

from app.survival_svm.settings import CONFIG_SECTION
from engine.app import App, AppState, app_state, BOTH, STATE_RUNNING, COORDINATOR
# This is the app instance, which holds various values and is used by the app states below
# You shouldn't access this app instance directly, just ignore it for now
from engine.survival_svm import ConfigFileState, ConfigWebState, CheckConfigSupplied, InitSplitManager, \
    BroadcastConfigState, ReceiveConfigState, ReadDataState, PreprocessDataState, SendDataAttributes, \
    InitOptimizer, ListenRequest, SendRequest, SetResponse, WriteResult, SendAggregatedDataAttributes, \
    SaveAggregatedDataAttributes, GeneratePredictions

app = App()
jsonpickle_numpy.register_handlers()


# This is the first (initial) state all app instances are in at the beginning
# By calling it 'initial' the FeatureCloud template engine knows that this state is the first one to go into automatically at the beginning
@app_state(app,
           'initial')  # The first argument is the name of the state ('initial'), the second specifies which roles are allowed to have this state (here BOTH)
class InitialState(AppState):

    def register(self):
        self.register_transition('check_config', BOTH)  # This is an iterative state

    def run(self) -> str or None:
        self.update(
            message=f'Initializing {"coordinator" if self.app.coordinator else "client"}',
            progress=0.01,
            state=STATE_RUNNING
        )
        self.app.internal['round'] = 0
        return 'check_config'


@app_state(app, 'check_config', section=CONFIG_SECTION,
           next_state_exists='read_config', next_state_missing='web_config')
class CheckConfigState(CheckConfigSupplied):
    pass


@app_state(app, 'read_config', section=CONFIG_SECTION,
           next_state_participant='receive_config', next_state_coordinator='broadcast_config')
class ReadConfigState(ConfigFileState):
    pass


@app_state(app, 'web_config',
           next_state_participant='receive_config', next_state_coordinator='broadcast_config')
class WebConfigState(ConfigWebState):
    pass


@app_state(app, 'broadcast_config', role=COORDINATOR, next_state='receive_config')
class BroadcastConfig(BroadcastConfigState):
    pass


@app_state(app, 'receive_config', next_state='init_split_manager')
class ReceiveConfig(ReceiveConfigState):
    pass


@app_state(app, 'init_split_manager', next_state='read_data')
class InitSplitManagerState(InitSplitManager):
    pass


@app_state(app, 'read_data', next_state='preprocess_data')
class ReadData(ReadDataState):
    pass


@app_state(app, 'preprocess_data', next_state='send_data_attributes')
class PreprocessData(PreprocessDataState):
    pass


@app_state(app, 'send_data_attributes',
           next_state_participant='read_aggr_data_attr', next_state_coordinator='initiate_opt')
class SendDataAttributesStep(SendDataAttributes):
    pass


@app_state(app, 'initiate_opt', role=COORDINATOR,
           next_state='send_aggr_data_attr')
class InitOptimizerStep(InitOptimizer):
    pass


@app_state(app, 'send_aggr_data_attr', role=COORDINATOR,
           next_state='read_aggr_data_attr')
class SendAggregatedDataAttributesStep(SendAggregatedDataAttributes):
    pass


@app_state(app, 'read_aggr_data_attr',
           next_state_participant='opt_listen', next_state_coordinator='opt_send_requests')
class SaveAggregatedDataAttributesStep(SaveAggregatedDataAttributes):
    pass


@app_state(app, 'opt_send_requests', role=COORDINATOR,
           next_state='opt_listen')
class SendRequestStep(SendRequest):
    pass


@app_state(app, 'opt_listen',
           next_state_participant='opt_listen', next_state_coordinator='opt_set_response',
           next_state_opt_finished='generate_results')
class ListenStep(ListenRequest):
    pass


@app_state(app, 'opt_set_response', role=COORDINATOR,
           next_state='opt_send_requests')
class SetResponseStep(SetResponse):
    pass


@app_state(app, 'generate_results',
           next_state='write_results')
class GeneratePredictionsState(GeneratePredictions):
    pass


@app_state(app, 'write_results',
           next_state=None)
class WriteResultState(WriteResult):
    pass
