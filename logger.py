import logging

# create a logger for the Meta-Cognition module
logger = logging.getLogger('Meta-Cognition')
logger.setLevel(logging.INFO)

# importing this logger for all files to make sure same logs are initilized in specific format
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# Add the logger to the global namespace for easy access
logger.info("Logger initialized for Meta-Cognition module.")
