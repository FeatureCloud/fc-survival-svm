FROM continuumio/miniconda3:latest

# Install your apt-get packages always like this to avoid cache problems
RUN apt-get update && apt-get install -y \
    supervisor nginx gcc
RUN pip3 install --upgrade pip

COPY server_config/supervisord.conf /supervisord.conf
COPY server_config/nginx /etc/nginx/sites-available/default
COPY server_config/docker-entrypoint.sh /entrypoint.sh

RUN conda install -c sebp scikit-survival
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
COPY . /app
RUN chmod u+x /app/main.py

EXPOSE 9000 9001

CMD ["sh", "/entrypoint.sh"]
