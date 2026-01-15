"""
Configuration module for queue-sample
Reads settings from environment variables
"""

import os

# Queue type: 'memory' or 'kafka'
QUEUE_TYPE = os.environ.get("QUEUE_TYPE", "memory").lower()

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "queue-sample-topic")
KAFKA_GROUP_ID = os.environ.get("KAFKA_GROUP_ID", "queue-sample-consumer-group")

# Kafka serialization format: 'protobuf' or 'json'
KAFKA_SERIALIZATION_FORMAT = os.environ.get("KAFKA_SERIALIZATION_FORMAT", "protobuf").lower()

# Kafka SSL/TLS configuration (optional)
KAFKA_SECURITY_PROTOCOL = os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_SSL_CAFILE = os.environ.get("KAFKA_SSL_CAFILE")
KAFKA_SSL_CERTFILE = os.environ.get("KAFKA_SSL_CERTFILE")
KAFKA_SSL_KEYFILE = os.environ.get("KAFKA_SSL_KEYFILE")

# Kafka consumer timeout configuration
# These control how long a consumer can process messages before being considered dead
KAFKA_MAX_POLL_INTERVAL_MS = int(os.environ.get("KAFKA_MAX_POLL_INTERVAL_MS", 300000))  # Default: 5 minutes
KAFKA_SESSION_TIMEOUT_MS = int(os.environ.get("KAFKA_SESSION_TIMEOUT_MS", 10000))  # Default: 10 seconds

NUM_WORKERS = int(os.environ.get("MIRIX_MEMORY_QUEUE_NUM_WORKERS", 1))
ROUND_ROBIN = os.environ.get("MIRIX_MEMORY_QUEUE_ROUND_ROBIN", "false").lower() in (
    "true",
    "1",
    "yes",
)

