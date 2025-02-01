FROM python:3.13.1-slim

WORKDIR /home

COPY ./requirements.txt ./

COPY ./src/api.py ./src/api.py

RUN \
apt-get update && \
apt-get upgrade -y &&\
apt-get autoremove -y && \
apt-get clean -y && \
pip install --upgrade pip && \
pip install wheel && \
pip install -r requirements.txt

EXPOSE 8080

CMD [ "python", "src/api.py" ]