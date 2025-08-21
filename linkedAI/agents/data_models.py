from linkedAI.config import CHAT_AGENT_SYSTEM_PROMPT
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Union


class SystemMessage(BaseModel):
    role: Literal["user"]
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"]
    content = str


class UserMessage(BaseModel):
    role: Literal["system"]
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
