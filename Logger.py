import logging
from typing import Optional


class Logger:
    """
    Logger class for application logging.

    This class configures and returns a logger instance that can be used throughout the application.
    The logger instance can write logs to both the console and a log file.

    :param name: The name of the logger.
    :param log_file: The optional path to the log file. If provided, logs will be written to this file.
    """

    def __init__(self, name: str, log_file: Optional[str] = None):
        """
        Initialize the Logger class.

        :param name: The name of the logger.
        :param log_file: The optional path to the log file. If provided, logs will be written to this file.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)

    def get_logger(self) -> logging.Logger:
        """
        Returns the configured logger instance.

        :return: The logger instance.
        """
        return self.logger
