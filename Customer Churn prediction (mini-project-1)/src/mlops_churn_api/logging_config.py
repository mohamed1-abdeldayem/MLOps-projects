import logging
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FORMATTER = logging.Formatter(LOG_FORMAT)


def create_logger(name: str, filename: str, filemode: str = "a") -> logging.Logger:
    """Create and configure a logger that writes messages to a log file.

    Args:
        name: Name of the logger.
        filename: Name of the log file.
        filemode: File opening mode, such as "a" for append or "w" for overwrite.

    Returns:
        A configured logging.Logger instance.
    """
    handler = logging.FileHandler(
        LOG_DIR / filename,
        encoding="utf-8",
        mode=filemode
    )
    handler.setLevel(logging.INFO)
    handler.setFormatter(FORMATTER)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    return logger


evaluation_logger = create_logger(
    "evaluation",
    "evaluations.log",
    "w"
)

