import logging

from bottle import Bottle
from jinja2 import Template

from .logic import logic

web_server = Bottle()

# CAREFUL: Do NOT perform any computation-related tasks inside these methods, nor inside functions called from them!
# Otherwise your app does not respond to calls made by the FeatureCloud system quickly enough
# Use the threaded loop in the app_flow function inside the file logic.py instead

TEMPLATE = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Survival SVM | Feature Cloud</title>
    <base href="/">
    <meta http-equiv="X-UA-Compatible" content="IE=edge"/>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Open+Sans" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link rel="icon" href="https://featurecloud.ai/assets/fc.ico">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bulma@0.9.1/css/bulma.min.css">
    <script src="https://kit.fontawesome.com/0206581977.js" crossorigin="anonymous"></script>
</head>
<body>
    <div class="main-wrapper">
      <main>
        <section class="hero is-link is-small">
          <div class="hero-body">
            <div class="container is-fluid">
              <section class="section" style="padding: 1rem 1rem;">
                <h1 class="title">
                  Survival SVM
                </h1>
              </section>
            </div>
          </div>
        </section>

        <div class="container is-fluid">
        <section class="section">
          <div class="card" style="margin-bottom: 15px;">
            <header class="card-header">
              <h2 class="card-header-title">Progress</h2>
            </header>
            <div class="card-content">
              {{ progress }}              
              <div><b>Iteration</b>: {{ iteration }}</div>
                {% if states is not none %}
                  {% for split, state in states.items() %}
                    <div><b>Split {{ split }}</b>: {{ state.state }}</div>
                  {% endfor %}
                {% endif %}
            </div>
          </div>
        </section>
        </div>
      </main>
    </div>
</body>
</html>
"""


@web_server.route('/')
def index():
    print(f'[WEB] GET /', flush=True)
    # return f'Progress: {logic.progress}\nIteration:{logic.iteration}\nStates: {logic.training_states}'
    template = Template(TEMPLATE)
    html = template.render(progress=logic.progress, iteration=logic.iteration, states=logic.training_states)
    logging.debug(html)
    return html
