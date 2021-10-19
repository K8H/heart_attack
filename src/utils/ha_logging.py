import logging

FORMAT = '%(asctime)-11s %(module)s.%(funcName)s %(levelname)s: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)