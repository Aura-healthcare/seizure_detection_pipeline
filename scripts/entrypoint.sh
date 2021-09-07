#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/dags:$(pwd)/src
pip install --user -r requirements.txt
airflow db init
airflow users create -r Admin -u admin -e admin@example.com -f admin -l user -p admin
export API_KEY_GRAFANA=$(curl -X POST -H "Content-Type: application/json" -d '{"name":"apikeycurl", "role": "Admin"}' http://admin:admin@grafana:3000/api/auth/keys)
airflow webserver &
airflow scheduler 
