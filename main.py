import logging

from bottle import Bottle

from api.http_ctrl import api_server
from api.http_web import web_server
from app import app

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [%(filename)s %(name)s %(funcName)s (%(lineno)d)]: %(message)s",
)

server = Bottle()

if __name__ == '__main__':
    app.register()
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    server.run(host='localhost', port=5000)
