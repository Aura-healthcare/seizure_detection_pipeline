# Getting Started
v 1.0

## Set up the Postgre database

1/ Install Postgre

```bash
foo@bar:sudo -u postgres createuser <username>
foo@bar: sudo -u postgres createdb <dbname>
sudo -u postgres psql
psql$ alter user <username> with encrypted password '<password>';
psql$ grant all privileges on database <dbname> to <username> ;

```

In the demo case:
 * dbname = postgres
 * username = postgres
 * password = postgres


 ### Upload the ECG data

 ```bash
 python3 ecg_loader.py --pg-host "localhost" --pg-port 5432 --pg-user <username> --pg-password <password> --pg-database <database> --filepath "00007633_s003_t007.edf" --exam "00007633_s003_t007"
 ```

### Upload the RR-intervals

```bash
python3 rr_intervals_loader.py --pg-host "localhost" --pg-port 5432 --pg-user <username> --pg-password <password> --pg-database <database> --filepath "rr_00007633_s003_t007.csv" --exam "00007633_s003_t007"
```

### Upload the annotations

```bash
python3 annotations_loader.py --pg-host "localhost" --pg-port 5432 --pg-user <username> --pg-password <password> --pg-database <database> --annotation-filename "00007633_s003_t007.tse_bi" --edf-filename "00007633_s003_t007.edf" --exam "00007633_s003_t007"
```

## 2/ Working with Docker

### a) Starting at least postgres container:

```bash
    $ source setup_env.sh
    $ docker-compose build postgres
    $ docker-compose up postgres -d
```

### b) Using Makefile to test

```bash
Make load_ecg
Make load_rr
Make load_annotations
```

## Set up the Grafana visualisation

## Possible improvements

* common functions to handle database creation
* common functions to load data
* better creation of database schema
* better parsing
* Taking into consideration environement variable to simplify credentials handling
