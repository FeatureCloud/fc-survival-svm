import logging

from bottle import Bottle
from jinja2 import Template

from .logic import logic

web_server = Bottle()


# CAREFUL: Do NOT perform any computation-related tasks inside these methods, nor inside functions called from them!
# Otherwise your app does not respond to calls made by the FeatureCloud system quickly enough
# Use the threaded loop in the app_flow function inside the file logic.py instead

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Survival SVM</title>
</head>
<body>
    <div><b>Progress</b>: {{ progress }}</div>
    <div><b>Iteration</b>: {{ iteration }}</div>
    {% if states is not none %}
        {% for split, state in states.items() %}
            <div><b>Split {{ split }}</b>: {{ state.state }}</div>
        {% endfor %}
    {% endif %}
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
