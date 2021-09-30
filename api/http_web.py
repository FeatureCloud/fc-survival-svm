import bottle
from bottle import Bottle

from app import app
from logic.splits import SplitManager

web_server = Bottle()


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
    return f'State: {app.current_state.name}'


@web_server.route('/web_config')
def web_config():
    print(f'[WEB] GET /')

    return f'NotImplemented'


@web_server.route('/during_training')
def during_training():
    print(f'[WEB] GET /')

    split_manager: SplitManager = app.internal.get('split_manager')

    return f"""
    <!doctype html>
    <html>
    <head>
    <title>Survival SVM</title>
    <meta name="description" content="Our first page">
    <meta name="keywords" content="html tutorial template">
    <link rel="stylesheet" href="static/bulma_0.9.3.min.css">
    </head>
    <body>
        <div class="main-wrapper">
            <main>
                <section class="hero is-link is-small">
                    <div class="hero-body">
                        <div class="container is-fluid">
                            <section class="section" style="padding: 1rem 1rem;">
                                <h1 class="title">
                                    Federated Survival SVM
                                </h1>
                            </section>
                        </div>
                    </div>
                </section>
                
                <div class="row">

                    <div class="columns">
                      <div class="column is-one-third">
                      
                          <div class="card" style="margin-bottom: 15px;">
                            <header class="card-header">
                                <h2 class="card-header-title">Global Options</h2>
                            </header>
                            <div class="card-content">
                                Text
                            </div>
                          </div>
                      
                      </div>
                      <div class="column">Auto</div>
                      <div class="column">Auto</div>
                    </div>
                
                </div>

            </main>
        </div>
    </body>
    </html>
    """

    return f'State: {app.current_state.name}\n' \
           f'Splits: {str(split_manager.data)}'
