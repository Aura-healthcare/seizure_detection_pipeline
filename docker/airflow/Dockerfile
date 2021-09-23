FROM apache/airflow:2.1.3

USER root
RUN apt-get update -y && apt-get install -y build-essential
COPY requirements.txt requirements.txt
USER airflow
RUN pip install --user -r requirements.txt
