"""
Configuration module for queue-sample
Reads settings from environment variables
"""
import os

# Queue type: 'memory' or 'kafka'
QUEUE_TYPE = os.environ.get('QUEUE_TYPE', 'memory').lower()

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.environ.get('KAFKA_TOPIC', 'queue-sample-topic')
KAFKA_GROUP_ID = os.environ.get('KAFKA_GROUP_ID', 'queue-sample-consumer-group')

# Kafka security configuration
KAFKA_SECURITY_PROTOCOL = os.environ.get('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT')  # PLAINTEXT or SSL
KAFKA_SSL_CAFILE = os.environ.get('KAFKA_SSL_CAFILE')  # Path to CA certificate
KAFKA_SSL_CERTFILE = os.environ.get('KAFKA_SSL_CERTFILE')  # Path to client certificate
KAFKA_SSL_KEYFILE = os.environ.get('KAFKA_SSL_KEYFILE')  # Path to client private key

# Kafka serialization format
KAFKA_SERIALIZATION_FORMAT = os.environ.get('KAFKA_SERIALIZATION_FORMAT', 'protobuf')  # protobuf or json
