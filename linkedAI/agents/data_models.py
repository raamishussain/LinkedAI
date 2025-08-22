from linkedAI.config import CHAT_AGENT_SYSTEM_PROMPT
from linkedAI.scraper.data_models import JobCard
from pydantic import BaseModel, Field
from typing import Any, Annotated, Literal, Optional, Union


class SystemMessage(BaseModel):
    role: Literal["system"]
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content = str


class UserMessage(BaseModel):
    role: Literal["user"]
    content: str


OpenAIMessage = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage],
    Field(discriminator="role"),
]


class ChatHistory(BaseModel):
    messages: list[OpenAIMessage] = [
        SystemMessage(content=CHAT_AGENT_SYSTEM_PROMPT)
    ]

    def to_messages(self) -> list[dict[str, str]]:
        """Function to output messages in OpenAI format"""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def append_message(self, message: OpenAIMessage):
        """Append a new message to the chat history"""
        self.messages.append(message)


class QueryArgs(BaseModel):
    """Arguments for querying the vector database (used in tool call)"""

    query: str
    n_results: int
    filters: Optional[dict[str, Any]]


class SearchResults(BaseModel):
    """Class containing list of jobs from querying the DB"""

    jobs: list[JobCard]
