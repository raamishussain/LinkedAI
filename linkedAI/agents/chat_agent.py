import json

from linkedAI.agents.agent import Agent
from linkedAI.agents.data_models import (
    AssistantMessage,
    ChatHistory,
    OpenAIMessage,
    QueryArgs,
    SearchResults,
    ToolMessage,
    UserMessage,
)
from linkedAI.agents.query_agent import QueryAgent
from linkedAI.config import CHROMA_COLLECTION, DEFAULT_MODEL, OPENAI_API_KEY
from openai import OpenAI
from typing import Any, Optional


class ChatAgent(Agent):
    """User facing agent class which facilitates basic chat and can call to
    other agents when needed"""

    name = "ChatAgent"
    model = DEFAULT_MODEL

    def __init__(self):
        self.query_agent = QueryAgent(
            chroma_path=CHROMA_COLLECTION, collection="linkedIn_jobs"
        )
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.chat_history = ChatHistory()

    def _tools(self) -> list[dict[str, Any]]:
        """Returns list of all available tools"""
        return [self.query_agent.tool_schema()]

    def chat(
        self, user_text: str
    ) -> tuple[OpenAIMessage, Optional[SearchResults]]:
        """Perform single chat interaction with the LLM

        Args:
            query: str
                User query to send to the model

        Returns:

        """

        self.chat_history.append_message(
            UserMessage(role="user", content=user_text)
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history.to_messages(),
            tools=self._tools(),
        )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        if not tool_calls:
            assistant_msg = AssistantMessage(
                role="assistant", content=choice.message.content
            )
            self.chat_history.append_message(assistant_msg)
            return assistant_msg, None

        self.chat_history.append_message(
            AssistantMessage(
                role="assistant",
                content=choice.message.content or "",
                tool_calls=[tc.model_dump() for tc in tool_calls]
                if tool_calls
                else None,
            )
        )

        args = json.loads(tool_calls[0].function.arguments)
        query_args = QueryArgs(**args)

        job_results = self.query_agent.query_vectorstore(query_args)

        self.chat_history.append_message(
            ToolMessage(
                role="tool",
                content=job_results.model_dump_json(),
                name="search_jobs",
                tool_call_id=tool_calls[0].id,
            )
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.chat_history.to_messages(),
        )

        reply = response.choices[0].message.content
        self.chat_history.append_message(
            AssistantMessage(
                role="assistant",
                content=reply,
            )
        )

        return reply, job_results
