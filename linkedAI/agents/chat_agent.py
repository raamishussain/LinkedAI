from linkedAI.agents import Agent, QueryAgent
from linkedAI.config import CHROMA_COLLECTION, DEFAULT_MODEL, OPENAI_API_KEY
from linkedAI.agents import (
    AssistantMessage,
    ChatHistory,
    OpenAIMessage,
    UserMessage,
)
from openai import OpenAI
from typing import Any


class ChatAgent(Agent):
    """User facing agent class which facilitates basic chat and can call to
    other agents when needed"""

    name = "ChatAgent"
    model = DEFAULT_MODEL

    def __init__(self):
        self.query_agent = QueryAgent(collection=CHROMA_COLLECTION)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.chat_history = ChatHistory()

    def _tools(self) -> list[dict[str, Any]]:
        """Returns list of all available tools"""
        return [self.query_agent.tool_schema()]

    def chat(self, user_text: str) -> OpenAIMessage:
        """Perform single chat interaction with the LLM

        Args:
            query: str
                User query to send to the model

        Returns:

        """

        self.chat_history.append_message(UserMessage(content=user_text))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history.to_messages(),
            tools=self._tools(),
        )

        choice = response.choices[0]
        tool_call = choice.message.tool_calls
        if not tool_call:
            assistant_msg = AssistantMessage(content=choice.message.content)
            self.chat_history.append_message(assistant_msg)
            return assistant_msg, None
