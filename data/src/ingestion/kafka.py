"""
Kafka consumer for transaction streaming (alternative to Event Hub)
"""

import json
import logging
from typing import Callable, Optional, Any
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from config.settings import settings

logger = logging.getLogger(__name__)


class KafkaTransactionConsumer:
    """
    Kafka consumer for real-time transaction streaming
    
    Usage:
        consumer = KafkaTransactionConsumer()
        consumer.start(on_message=process_transaction)
    """

    def __init__(self):
        """Initialize Kafka consumer"""
        self.bootstrap_servers = settings.kafka.bootstrap_servers
        self.topic = settings.kafka.topic
        self.group_id = settings.kafka.group_id
        self.consumer: Optional[Any] = None
        self.is_running = False

    def connect(self) -> None:
        """Establish connection to Kafka"""
        try:
            self.consumer = KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                consumer_timeout_ms=settings.kafka.consumer_timeout_ms,
                max_poll_records=100
            )
            logger.info(f"Connected to Kafka topic: {self.topic}")
        except KafkaError as e:
            logger.error(f"Failed to connect to Kafka: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Close Kafka connection"""
        if self.consumer:
            self.consumer.close()
            self.is_running = False
            logger.info("Disconnected from Kafka")

    def start(self, on_message: Callable[[dict], None]) -> None:
        """
        Start consuming messages from Kafka
        
        Args:
            on_message: Callback function to process each message
        """
        if not self.consumer:
            self.connect()

        self.is_running = True

        try:
            for message in self.consumer:
                if message.value:
                    try:
                        on_message(message.value)
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Error in Kafka consumer: {str(e)}")
        finally:
            self.disconnect()

    def get_topics(self) -> dict:
        """Get list of available topics"""
        if not self.consumer:
            self.connect()
        
        try:
            return self.consumer.topics()
        except Exception as e:
            logger.error(f"Failed to get topics: {str(e)}")
            return {}

    def get_partitions(self, topic: str) -> list:
        """Get partitions for a specific topic"""
        if not self.consumer:
            self.connect()
        
        try:
            return list(self.consumer.partitions_for_topic(topic))
        except Exception as e:
            logger.error(f"Failed to get partitions for {topic}: {str(e)}")
            return []
