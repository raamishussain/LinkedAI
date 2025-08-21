import logging


class Agent:
    """Abstract class for an agent"""

    name: str = ""

    def log(self, message):
        """Log the message from a particular agent"""
        message = f"[{self.name}] {message}"
        logging.info(message)
