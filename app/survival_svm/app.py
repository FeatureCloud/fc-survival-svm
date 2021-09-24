from engine.app import App, AppState, app_state, BOTH, STATE_RUNNING

# This is the app instance, which holds various values and is used by the app states below
# You shouldn't access this app instance directly, just ignore it for now
from engine.survival_svm import ConfigFileState, ConfigWebState, CheckConfigSupplied

app = App()


# This is the first (initial) state all app instances are in at the beginning
# By calling it 'initial' the FeatureCloud template engine knows that this state is the first one to go into automatically at the beginning
@app_state(app, 'initial')  # The first argument is the name of the state ('initial'), the second specifies which roles are allowed to have this state (here BOTH)
class InitialState(AppState):

    def register(self):
        self.register_transition('check_config', BOTH)  # This is an iterative state

    def run(self) -> str or None:
        self.update(
            message=f'Initializing {"coordinator" if self.app.coordinator else "client"}',
            progress=0.01,
            state=STATE_RUNNING
        )
        return 'check_config'


CONFIG_SECTION = 'fc_survival_svm'


@app_state(app, 'check_config', section=CONFIG_SECTION, next_state_exists='read_config',
           next_state_missing='web_config')
class CheckConfigState(CheckConfigSupplied):
    pass


@app_state(app, 'read_config', section=CONFIG_SECTION, next_state='read_data')
class ReadConfigState(ConfigFileState):
    pass


@app_state(app, 'web_config', next_state='read_data')
class WebConfigState(ConfigWebState):
    pass


@app_state(app, 'read_data')
class ReadDataState(AppState):

    def run(self):
        self.app.log(self.app.internal['config'])
        return None
