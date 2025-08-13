import os
import logging.config
import yaml
from datetime import datetime

def setup_logging(output_path, config_path='logging_config.yaml', default_level=logging.INFO):
    """
    Set up logging using a YAML config file. A dynamic log file is created in output_path/log.
    """
    log_dir = os.path.join(output_path, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if 'handlers' in config and 'file' in config['handlers']:
            config['handlers']['file']['filename'] = log_file
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
