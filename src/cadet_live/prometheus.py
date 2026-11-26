import logging
from datetime import datetime, timedelta
from prometheus_api_client import PrometheusConnect

logger = logging.getLogger(__name__)

class Client:
    def __init__(self, config: dict) -> None:
        logger.debug("Init Client with", config)
        self.host = config["host"] or "127.0.0.1"
        self.port = config["port"] or 9090
        self.mapping = config.get("input_mapping")
        logger.info("Create Client with info: ", self)

    def get_data(self):
        prom = PrometheusConnect(url=self.host, disable_ssl=True)
    
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=2)
    
        for measurement_mapping in self.mapping:
        # Get metric data over time range
            print(type(measurement_mapping))
            print(measurement_mapping.get("name"))
            range_data = prom.custom_query_range(
                query=measurement_mapping.get("name"),
                start_time=start_time,  # datetime object
                end_time=end_time,  
                step='15s',
            )
            for value in range_data[0].get("values"):
                value[0] = datetime.utcfromtimestamp(value[0])