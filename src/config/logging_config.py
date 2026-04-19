"""
Logging Configuration Module
Sets up structured logging for the handwriting recognition API
"""

import io
import logging
import logging.handlers
import os
import sys
from datetime import datetime


def setup_logging(log_dir='logs', log_name='app'):
    """
    Setup logging configuration with file and console handlers
    
    Args:
        log_dir (str): Directory to store log files
        log_name (str): Name of the logger
    
    Returns:
        logging.Logger: Configured logger instance
    """
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Log file path
    log_file = os.path.join(log_dir, f'{log_name}.log')
    
    # File handler with rotation (10MB per file, keep 10 backups)
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    fh.setLevel(logging.DEBUG)
    
    # Console handler (INFO level and above)
    stdout_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    ch = logging.StreamHandler(stdout_stream)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# Custom Exception Classes
class ValidationError(Exception):
    """Raised when input validation fails"""
    pass


class ModelError(Exception):
    """Raised when model inference fails"""
    pass


class PreprocessingError(Exception):
    """Raised when image preprocessing fails"""
    pass


class SpellcheckError(Exception):
    """Raised when spellcheck fails"""
    pass
