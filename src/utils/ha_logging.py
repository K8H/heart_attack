import logging
import os
from pathlib import Path


class Logger(object):

    def __init__(self, filename_dir: str):
        Path(filename_dir).touch()
        FORMAT = '%(asctime)-11s %(module)s.%(funcName)s %(levelname)s: %(message)s'
        logging.basicConfig(filename=filename_dir, format=FORMAT, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
