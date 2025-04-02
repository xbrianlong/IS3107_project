FROM apache/airflow:2.10.5

# Install additional Python libraries
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
