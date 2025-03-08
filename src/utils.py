# utils.py
import logging
import os

def configure_logging(log_level=logging.INFO):
    """
    Configure and return a logger for the project.
    Logs are written to both the console and a file.
    :param log_level: Logging level (e.g., logging.INFO)
    :return: Configured logger instance.
    """
    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("ImageClassification")
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers if configure_logging is called multiple times.
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler("logs/app.log")
        fh.setLevel(log_level)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
