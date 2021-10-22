from time import sleep

import bottle
import jinja2

from app import app
from logic.config import Config
from logic.splits import SplitManager

web_server = bottle.Bottle()


# CAREFUL: Do NOT perform any computation-related tasks inside these methods, nor inside functions called from them!
# Otherwise your app does not respond to calls made by the FeatureCloud system quickly enough
# Use the threaded loop in the app_flow function inside the file logic.py instead


@web_server.route('/')
def index():
    print(f'[WEB] GET /')

    if app.current_state.name == 'web_config':
        bottle.redirect('/web_config')
    elif app.current_state.name in ['opt_send_requests', 'opt_listen', 'opt_set_response']:
        bottle.redirect('/during_training')
    elif app.current_state.name in ['write_results', 'generate_results']:
        bottle.redirect('/during_training')
    return f'State: {app.current_state.name}'


@web_server.route('/web_config')
def web_config():
    print(f'[WEB] GET /web_config')

    is_coordinator = app.coordinator
    min_samples = Config.DEFAULT_MIN_SAMPLES

    templateLoader = jinja2.FileSystemLoader(searchpath="/app/templates")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template('web_config_form.html')
    html = template.render(is_coordinator=is_coordinator, min_samples=min_samples)
    return html


@web_server.post('/web_config_apply')
def web_config_apply():
    print(f'[WEB] GET /web_config_apply')

    web_config = bottle.request.forms.decode()

    is_coordinator = app.coordinator

    config: Config = Config.from_web(web_config, is_coordinator)
    app.internal['config'] = config

    sleep(3)

    return bottle.redirect('/')


@web_server.route('/during_training')
def during_training():
    print(f'[WEB] GET /during_training')

    split_manager: SplitManager = app.internal.get('split_manager')

    templateLoader = jinja2.FileSystemLoader(searchpath="/app/templates")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template('info.html')
    html = template.render(round=app.internal.get('round'), split_manager=split_manager)
    return html
