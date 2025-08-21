from linkedAI.agents import Agent, QueryAgent
from linkedAI.config import CHROMA_COLLECTION, OPENAI_API_KEY
from linkedAI.agents import OpenAIMessage, ChatHistory
from openai import OpenAI


class ChatAgent(Agent):
    """User facing agent class which facilitates basic chat and can call to
    other agents when needed"""

    name = "ChatAgent"

    def __init__(self):
        self.query_agent = QueryAgent(collection=CHROMA_COLLECTION)
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def chat(self, query: str) -> OpenAIMessage:
        """Perform single chat interaction with the LLM

        Args:
            query: str
                User query to send to the model

        Returns:
        """
