import logging


class Agent:
    """Abstract class for an agent"""

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{self.name}")

    def log(self, message: str, level: str = "info", **kwargs):
        """Log message with specified level"""
        level_map = {
            "debug": self.logger.debug,
            "info": self.logger.info,
            "warning": self.logger.warning,
            "error": self.logger.error,
            "critical": self.logger.critical,
            "exception": self.logger.exception,
        }

        log_func = level_map.get(level.lower(), self.logger.info)
        formatted_message = f"[{self.name}] {message}"
        log_func(formatted_message, extra=kwargs)
