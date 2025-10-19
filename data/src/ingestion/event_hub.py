"""
Azure Event Hub consumer for real-time transaction streaming
"""

import json
import logging
from typing import Optional, Callable, Any

# Handle optional Azure Event Hub dependency (not available on Python 3.11 by default)
try:
    from azure.eventhub import EventHubConsumerClient
    from azure.eventhub.exceptions import EventHubError
    EVENTHUB_AVAILABLE = True
except ImportError:
    EVENTHUB_AVAILABLE = False
    EventHubConsumerClient = None
    class EventHubError(Exception):
        """Placeholder for EventHubError when azure-eventhub not available"""
        pass

from config.settings import settings

logger = logging.getLogger(__name__)


class EventHubConsumer:
    """
    Consumer for Azure Event Hub - ingests real-time transaction events
    
    Usage:
        consumer = EventHubConsumer()
        consumer.start(on_event_received=process_transaction)
    """

    def __init__(self, fully_qualified_namespace: Optional[str] = None):
        """
        Initialize Event Hub consumer
        
        Args:
            fully_qualified_namespace: Event Hub namespace URL
                                      Defaults to settings if not provided
        """
        self.connection_string = settings.azure.event_hub_connection_string
        self.event_hub_name = settings.azure.event_hub_name
        self.client = None  # Will be EventHubConsumerClient if available
        self.is_running = False
        
        if not EVENTHUB_AVAILABLE:
            logger.warning(
                "azure-eventhub not available. Install with: pip install -r requirements-azure.txt"
            )

    def connect(self) -> None:
        """Establish connection to Event Hub"""
        if not EVENTHUB_AVAILABLE:
            raise RuntimeError(
                "azure-eventhub not installed. "
                "Install with: pip install -r requirements-azure.txt"
            )
        
        try:
            self.client = EventHubConsumerClient.from_connection_string(
                conn_str=self.connection_string,
                consumer_group="$Default",
                eventhub_name=self.event_hub_name
            )
            logger.info(f"Connected to Event Hub: {self.event_hub_name}")
        except EventHubError as e:
            logger.error(f"Failed to connect to Event Hub: {str(e)}")
            raise

    def disconnect(self) -> None:
        """Close Event Hub connection"""
        if self.client:
            self.client.close()
            self.is_running = False
            logger.info("Disconnected from Event Hub")

    def start(
        self,
        on_event_received: Callable[[dict], None],
        partition_id: Optional[str] = None,
        starting_position: int = "-1"  # -1 = from end, "@latest" = latest event
    ) -> None:
        """
        Start consuming events from Event Hub
        
        Args:
            on_event_received: Callback function to process each event
            partition_id: Specific partition to consume from (optional)
            starting_position: Where to start reading (-1 for end, 0+ for offset)
        """
        if not self.client:
            self.connect()

        self.is_running = True

        def on_partition_init(partition_context):
            """Initialize partition"""
            logger.info(f"Partition initialized: {partition_context.partition_id}")

        def on_error(partition_context, error):
            """Handle partition errors"""
            logger.error(
                f"Error on partition {partition_context.partition_id}: {str(error)}"
            )

        def on_event(partition_context, event):
            """Process incoming event"""
            try:
                if event:
                    event_data = json.loads(event.body_as_str())
                    on_event_received(event_data)
                    partition_context.update_checkpoint(event)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse event JSON: {str(e)}")
            except Exception as e:
                logger.error(f"Error processing event: {str(e)}")
                on_error(partition_context, e)

        try:
            self.client.receive_batch(
                on_event=on_event,
                on_error=on_error,
                on_partition_init=on_partition_init,
                partition_id=partition_id,
                starting_position=starting_position,
                max_wait_time=1  # seconds
            )
        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user")
        except Exception as e:
            logger.error(f"Error in event consumer: {str(e)}")
        finally:
            self.disconnect()

    def get_partition_ids(self) -> list[str]:
        """Get all available partition IDs"""
        if not self.client:
            self.connect()
        
        try:
            properties = self.client.get_eventhub_properties()
            partition_ids = properties["partition_ids"]
            logger.info(f"Available partitions: {partition_ids}")
            return partition_ids
        except Exception as e:
            logger.error(f"Failed to get partition IDs: {str(e)}")
            return []
