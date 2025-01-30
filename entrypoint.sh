#!/bin/bash

# Wait for MySQL to be ready
echo "Waiting for MySQL to start..."
while ! nc -z mysql 3306; do
    sleep 1
done
echo "MySQL is up and running!"

# Initialize the Airflow database if not already initialized
if [ ! -f "/opt/airflow/airflow.db_initialized" ]; then
    echo "Initializing Airflow database..."
    airflow db migrate
    touch /opt/airflow/airflow.db_initialized
else
    echo "Airflow database already initialized."
fi

# Start the Airflow webserver and scheduler
airflow webserver -p 8080 & airflow scheduler
