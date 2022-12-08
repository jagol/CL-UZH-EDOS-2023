import argparse
import logging
import os
import sys


def get_logger(level='main') -> logging.Logger:
    log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    root_logger = logging.getLogger(level)
    file_handler = logging.FileHandler(os.path.join('logs', f'{level}_logs.txt'))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel('INFO')
    return root_logger
