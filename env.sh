# InfluxDB env vars
export INFLUXDB_USERNAME=admin
export INFLUXDB_PASSWORD=auraadmin
export INFLUXDB_DATABASE=hackathon

# Grafana env vars
export GRAFANA_USERNAME=admin
export GRAFANA_PASSWORD=admin
export POSTGRES_DATABASE_GRAFANA=grafana

# PostgreSQL env vars
export POSTGRES_HOST_URL=postgres
export POSTGRES_DATABASE=postgres
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres

# Airflow env vars
export POSTGRES_DATABASE_AIRFLOW=airflow

# MLFlow env vars
export POSTGRES_DATABASE_MLFLOW=mlflow

# IDs
export AIRFLOW_UID=$(id -u)
export AIRFLOW_GID=$(id -g)

# Ports
export POSTGRES_PORT=5432
export INFLUXDB_PORT=8086
export AIRFLOW_PORT=8080
export GRAFANA_PORT=3000
export MLFLOW_PORT=5000
