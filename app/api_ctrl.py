import logging
import json
import time

from bottle import Bottle, request

from .logic import logic

api_server = Bottle()


# CAREFUL: Do NOT perform any computation-related tasks inside these methods, nor inside functions called from them!
# Otherwise your app does not respond to calls made by the FeatureCloud system quickly enough
# Use the threaded loop in the app_flow function inside the file logic.py instead


@api_server.post('/setup')
def ctrl_setup():
    time.sleep(1)
    logging.debug(f'[CTRL] POST /setup')
    payload = request.json
    logic.handle_setup(payload['id'], payload['master'], payload['clients'])
    return ''


@api_server.get('/status')
def ctrl_status():
    logging.debug(f'[CTRL] GET /status available={logic.status_available} finished={logic.status_finished}')
    return json.dumps({
        'available': logic.status_available,
        'finished': logic.status_finished,
    })


@api_server.route('/data', method='GET')
def ctrl_data_out():
    data = logic.handle_outgoing()
    logging.debug(f'[CTRL] GET /data data={data}')
    return data


@api_server.route('/data', method='POST')
def ctrl_data_in():
    data = request.body
    logging.debug(f'[CTRL] POST /data data={data}')
    logic.handle_incoming(data)
    return ''
