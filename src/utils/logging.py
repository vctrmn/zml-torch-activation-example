import os
from logging import INFO, Formatter, Logger, StreamHandler, getLogger


class CustomFormatter(Formatter):
    def format(self, record):
        record.name = record.name.split(".")[-1]
        return super().format(record)


def get_logger(name: str) -> Logger:
    # Create a custom logger
    logger = getLogger(name)
    log_level = os.environ.get("LOG_LEVEL", None)

    # Create a console handler and set its level and formatter
    console_handler = StreamHandler()

    # Set its level
    if log_level is not None and str(log_level).upper() in ("DEBUG", "INFO"):
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())
        console_handler.setLevel(log_level.upper())
    else:
        logger.setLevel(INFO)
        logger.root.setLevel(INFO)
        console_handler.setLevel(INFO)

    # Set its formatter
    formatter = CustomFormatter("%(asctime)s - [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger
