# Start with an official TensorFlow Serving image
FROM gcr.io/tfx-oss-public/tfx:1.14.0

# Set the working directory inside the container
WORKDIR /app

# Update apt-get package lists
RUN apt-get update --allow-releaseinfo-change

# Install build-essential package
RUN apt-get install -y build-essential

# Install GCC (GNU Compiler Collection)
RUN apt-get install -y gcc

# Install G++ (GNU C++ compiler)
RUN apt-get install -y g++

# Install Python3 development headers
RUN apt-get install -y python3-dev

# Clean up unnecessary files to reduce image size
RUN apt-get clean

# Install system dependencies for MySQL
RUN apt-get update && apt-get install -y \
    libmysqlclient-dev \
    libssl-dev \
    python3-dev \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Install mysqlclient
RUN pip install mysqlclient

# Install Python dependencies
RUN pip install tfx
RUN pip install deap

# Set environment variables for MySQL client
ENV MYSQLCLIENT_CFLAGS="-I/usr/include/mysql"
ENV MYSQLCLIENT_LDFLAGS="-L/usr/lib/mysql"

# Install PyMySQL as an alternative to MySQLdb
RUN pip install PyMySQL

# Install Airflow with MySQL integration
RUN pip install apache-airflow[mysql]

# Optionally install airflow with tfx integration
RUN pip install tfx[airflow]

# Ensure the Airflow configuration points to PyMySQL for MySQL
RUN mkdir /etc/airflow
RUN echo "[core]\n\
sql_alchemy_conn = mysql+pymysql://root:rootpassword@mysql:3306/airflow_db" >> /etc/airflow/airflow.cfg

COPY ./dags /opt/airflow/dags

# Copy the rest of the project files into the container
COPY . /app

# Set up environment variables
ENV PYTHONPATH=/app

# Expose the port for the Airflow UI
EXPOSE 8080

# Use an entrypoint script to initialize and start Airflow
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]