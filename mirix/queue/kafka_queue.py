"""
Kafka queue implementation
Requires kafka-python and protobuf libraries to be installed
Uses Google Protocol Buffers for message serialization
"""
import logging
from typing import Optional

from mirix.queue.queue_interface import QueueInterface
from mirix.queue.message_pb2 import QueueMessage

logger = logging.getLogger(__name__)


class KafkaQueue(QueueInterface):
    """Kafka-based queue implementation with SSL/mTLS support and flexible serialization"""
    
    def __init__(
        self,
        bootstrap_servers: str,
        topic: str,
        group_id: str,
        security_protocol: str = "PLAINTEXT",
        ssl_cafile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        ssl_keyfile: Optional[str] = None,
        serialization_format: str = "protobuf"
    ):
        """
        Initialize Kafka producer and consumer with SSL/mTLS and flexible serialization
        
        Args:
            bootstrap_servers: Kafka broker address(es)
            topic: Kafka topic name
            group_id: Consumer group ID
            security_protocol: Security protocol - "PLAINTEXT" (default) or "SSL" for mTLS
            ssl_cafile: Path to CA certificate file (required for SSL)
            ssl_certfile: Path to client certificate file (required for SSL)
            ssl_keyfile: Path to client private key file (required for SSL)
            serialization_format: Serialization format - "protobuf" (default) or "json"
        """
        logger.debug(
            "Initializing Kafka queue: servers=%s, topic=%s, group=%s, security=%s, format=%s",
            bootstrap_servers, topic, group_id, security_protocol, serialization_format
        )
        
        try:
            from kafka import KafkaProducer, KafkaConsumer
        except ImportError:
            logger.error("kafka-python not installed")
            raise ImportError(
                "kafka-python is required for Kafka support. "
                "Install it with: pip install queue-sample[kafka]"
            )
        
        # Validate SSL configuration
        if security_protocol == "SSL":
            if not all([ssl_cafile, ssl_certfile, ssl_keyfile]):
                raise ValueError(
                    "SSL security protocol requires ssl_cafile, ssl_certfile, and ssl_keyfile"
                )
            logger.info("Using SSL/mTLS for Kafka connection")
            logger.debug("SSL config: ca=%s, cert=%s, key=%s", ssl_cafile, ssl_certfile, ssl_keyfile)
        
        self.topic = topic
        self.serialization_format = serialization_format
        
        # Choose serialization format
        if serialization_format == "json":
            # JSON serialization for Event Bus compatibility
            import json
            from google.protobuf.json_format import MessageToDict, ParseDict
            
            def json_serializer(message: QueueMessage) -> bytes:
                """
                Serialize QueueMessage to JSON format
                
                Args:
                    message: QueueMessage protobuf to serialize
                    
                Returns:
                    JSON bytes
                """
                message_dict = MessageToDict(message)
                return json.dumps(message_dict).encode('utf-8')
            
            def json_deserializer(serialized_msg: bytes) -> QueueMessage:
                """
                Deserialize JSON to QueueMessage
                
                Args:
                    serialized_msg: JSON bytes
                    
                Returns:
                    QueueMessage protobuf object
                """
                message_dict = json.loads(serialized_msg.decode('utf-8'))
                msg = QueueMessage()
                ParseDict(message_dict, msg)
                return msg
            
            value_serializer = json_serializer
            value_deserializer = json_deserializer
            logger.info("Using JSON serialization for Kafka messages")
        else:
            # Protobuf serialization (default)
            def protobuf_serializer(message: QueueMessage) -> bytes:
                """
                Serialize QueueMessage to Protocol Buffer format
                
                Args:
                    message: QueueMessage protobuf to serialize
                    
                Returns:
                    Serialized protobuf bytes
                """
                return message.SerializeToString()
            
            def protobuf_deserializer(serialized_msg: bytes) -> QueueMessage:
                """
                Deserialize Protocol Buffer message to QueueMessage
                
                Args:
                    serialized_msg: Serialized protobuf bytes
                    
                Returns:
                    QueueMessage protobuf object
                """
                msg = QueueMessage()
                msg.ParseFromString(serialized_msg)
                return msg
            
            value_serializer = protobuf_serializer
            value_deserializer = protobuf_deserializer
            logger.info("Using Protobuf serialization for Kafka messages")
        
        # Build Kafka configuration
        producer_config = {
            'bootstrap_servers': bootstrap_servers,
            'key_serializer': lambda k: k.encode('utf-8'),  # Encode partition key to bytes
            'value_serializer': value_serializer,
            'security_protocol': security_protocol
        }
        
        consumer_config = {
            'bootstrap_servers': bootstrap_servers,
            'group_id': group_id,
            'value_deserializer': value_deserializer,
            'auto_offset_reset': 'earliest',  # Start from beginning if no offset exists
            'enable_auto_commit': True,
            'consumer_timeout_ms': 1000,  # Timeout for polling
            'security_protocol': security_protocol
        }
        
        # Add SSL configuration if using SSL
        if security_protocol == "SSL":
            ssl_config = {
                'ssl_cafile': ssl_cafile,
                'ssl_certfile': ssl_certfile,
                'ssl_keyfile': ssl_keyfile,
                'ssl_check_hostname': True
            }
            producer_config.update(ssl_config)
            consumer_config.update(ssl_config)
        
        # Initialize Kafka producer with SSL/mTLS support
        logger.debug("Creating Kafka producer with config: %s", {k: v for k, v in producer_config.items() if 'serializer' not in k})
        self.producer = KafkaProducer(**producer_config)
        
        # Initialize Kafka consumer with SSL/mTLS support
        logger.debug("Creating Kafka consumer with config: %s", {k: v for k, v in consumer_config.items() if 'deserializer' not in k})
        self.consumer = KafkaConsumer(topic, **consumer_config)
    
    def put(self, message: QueueMessage) -> None:
        """
        Send a message to Kafka topic with user_id as partition key.
        
        This ensures all messages for the same user go to the same partition,
        guaranteeing single-worker processing and message ordering per user.
        
        Implementation:
        - Uses user_id (or actor.id as fallback) as partition key
        - Kafka assigns partition via: hash(key) % num_partitions
        - Consumer group ensures only one worker per partition
        - Result: Same user always processed by same worker (no race conditions)
        
        Args:
            message: QueueMessage protobuf message to send
        """
        # Extract user_id as partition key (fallback to actor.id if not present)
        partition_key = message.user_id if message.user_id else message.actor.id
        
        logger.debug(
            "Sending message to Kafka topic %s: agent_id=%s, partition_key=%s",
            self.topic, message.agent_id, partition_key
        )
        
        # Send message with partition key - ensures consistent partitioning
        # Kafka will route this to: partition = hash(partition_key) % num_partitions
        future = self.producer.send(
            self.topic,
            key=partition_key,  # Partition key for consistent routing
            value=message
        )
        future.get(timeout=10)  # Wait up to 10 seconds for confirmation
        
        logger.debug("Message sent to Kafka successfully with partition key: %s", partition_key)
    
    def get(self, timeout: Optional[float] = None) -> QueueMessage:
        """
        Retrieve a message from Kafka
        
        Args:
            timeout: Not used for Kafka (uses consumer_timeout_ms instead)
            
        Returns:
            QueueMessage protobuf message from Kafka
            
        Raises:
            StopIteration: If no message available
        """
        logger.debug("Polling Kafka topic %s for messages", self.topic)
        
        # Poll for messages
        for message in self.consumer:
            logger.debug("Retrieved message from Kafka: agent_id=%s", message.value.agent_id)
            return message.value
        
        # If no message received, raise exception (similar to queue.Empty)
        logger.debug("No message available from Kafka")
        raise StopIteration("No message available")
    
    def close(self) -> None:
        """Close Kafka producer and consumer connections"""
        logger.info("Closing Kafka connections")
        
        if hasattr(self, 'producer'):
            self.producer.close()
            logger.debug("Kafka producer closed")
        if hasattr(self, 'consumer'):
            self.consumer.close()
            logger.debug("Kafka consumer closed")

