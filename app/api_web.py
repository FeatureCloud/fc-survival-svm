import logging
import time

import bottle
import jinja2

from .logic import logic

web_server = bottle.Bottle()

# CAREFUL: Do NOT perform any computation-related tasks inside these methods, nor inside functions called from them!
# Otherwise your app does not respond to calls made by the FeatureCloud system quickly enough
# Use the threaded loop in the app_flow function inside the file logic.py instead


@web_server.route('/')
def index():
    logging.debug(f'[WEB] GET /')
    # return f'Progress: {logic.progress}\nIteration:{logic.iteration}\nStates: {logic.training_states}'
    templateLoader = jinja2.FileSystemLoader(searchpath="/app/templates")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template('info.html')
    html = template.render(progress=logic.progress, iteration=logic.iteration, states=logic.training_states)
    return html
