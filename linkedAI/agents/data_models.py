from linkedAI.config import CHAT_AGENT_SYSTEM_PROMPT
from linkedAI.scraper.data_models import JobCard
from pydantic import BaseModel, Field
from typing import Any, Annotated, Literal, Optional, Union


# Data models related to OpenAI message format
class SystemMessage(BaseModel):
    role: Literal["system"]
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: str = ""
    tool_calls: Optional[list[dict[str, Any]]] = None


class UserMessage(BaseModel):
    role: Literal["user"]
    content: str


class ToolMessage(BaseModel):
    role: Literal["tool"]
    content: str
    tool_call_id: str
    name: Optional[str] = None


OpenAIMessage = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator="role"),
]


class ChatHistory(BaseModel):
    messages: list[OpenAIMessage] = [
        SystemMessage(role="system", content=CHAT_AGENT_SYSTEM_PROMPT)
    ]

    def to_messages(self) -> list[dict[str, str]]:
        """Function to output messages in OpenAI format"""
        output = []
        for m in self.messages:
            if isinstance(m, AssistantMessage):
                d = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    d["tool_calls"] = m.tool_calls
                output.append(d)
            elif isinstance(m, ToolMessage):
                d = {
                    "role": "tool",
                    "content": m.content,
                    "tool_call_id": m.tool_call_id,
                }
                if m.name:
                    d["name"] = m.name
                output.append(d)
            else:
                output.append({"role": m.role, "content": m.content})

        return output

    def append_message(self, message: OpenAIMessage):
        """Append a new message to the chat history"""
        self.messages.append(message)


# Data models for QueryAgent's tool call
class QueryArgs(BaseModel):
    """Arguments for querying the vector database (used in tool call)"""

    query: str
    n_results: int


class SearchResults(BaseModel):
    """Class containing list of jobs from querying the DB"""

    jobs: list[JobCard]


# Data models for ResumeAgent's tool calls
class ResumeMatchArgs(BaseModel):
    """Arguments"""

    jobs: SearchResults


class ResumeMatchResult(BaseModel):
    """"""

    best_match_id: int
    reasoning: str


class ResumeTweakArgs(BaseModel):
    """Arguments for"""

    job_description: str


class ResumeTweakResult(BaseModel):
    suggestions: str
