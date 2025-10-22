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


