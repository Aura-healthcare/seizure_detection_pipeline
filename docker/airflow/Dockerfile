FROM apache/airflow:2.1.3

COPY requirements.txt requirements.txt
USER root
RUN apt-get update
RUN apt-get -y install build-essential
USER airflow
RUN pip install --user -r requirements.txt
