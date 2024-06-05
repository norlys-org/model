# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.9.6

FROM python:${PYTHON_VERSION}-slim

LABEL fly_launch_runtime="flask"

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY app.py /code/app.py
COPY config.py /code/config.py
COPY app/ /code/app
COPY config/ /code/config
COPY data/month /code/data/month
COPY data/quantiles.json /code/data/quantiles.json
COPY data/archive_update_date.txt /code/data/archive_update_date.txt
COPY data/0m-model.joblib /code/data/0m-model.joblib

EXPOSE 8080

CMD [ "python3", "app.py"]