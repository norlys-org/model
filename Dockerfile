# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.9.6

FROM python:${PYTHON_VERSION}-slim

LABEL fly_launch_runtime="flask"

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY app.py /code/app.py

EXPOSE 80

CMD [ "python3", "app.py" ]