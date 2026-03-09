# logging_config.py
import logging
import logging.config
import yaml
import os
pPath = os.path.split(os.path.realpath(__file__))[0]

def setup_logging(default_path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG', log_filename='app.log'):
    """
    Setup logging configuration
    """
    path = os.path.join(pPath,default_path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
            config['handlers']['file_handler']['filename'] = os.path.join(pPath,log_filename)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
