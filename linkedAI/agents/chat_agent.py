import json

from linkedAI.agents.agent import Agent
from linkedAI.agents.data_models import (
    AssistantMessage,
    ChatHistory,
    OpenAIMessage,
    QueryArgs,
    ResumeMatchArgs,
    ResumeTweakArgs,
    SearchResults,
    ToolMessage,
    UserMessage,
)
from linkedAI.agents.query_agent import QueryAgent
from linkedAI.agents.resume_agent import ResumeAgent
from linkedAI.config import (
    CHROMA_COLLECTION,
    DEFAULT_MODEL,
    OPENAI_API_KEY,
    RESUME_PATH,
)
from openai import OpenAI
from pathlib import Path
from typing import Any, Optional


class InvalidToolError(Exception):
    """Exception raised for invalid tool call"""

    pass


class ChatAgent(Agent):
    """User facing agent class which facilitates basic chat and can call to
    other agents when needed"""

    def __init__(self):

        super().__init__("ChatAgent")

        self.model = DEFAULT_MODEL
        self.client = OpenAI(api_key=OPENAI_API_KEY)

        self.query_agent = QueryAgent(
            openai_client=self.client,
            chroma_path=CHROMA_COLLECTION,
            collection="linkedIn_jobs",
        )
        self.resume_agent = ResumeAgent(
            openai_client=self.client,
            resume_path=Path(RESUME_PATH),
        )

        self.chat_history = ChatHistory()

    def _tools(self) -> list[dict[str, Any]]:
        """Returns list of all available tools"""
        return [
            self.query_agent.tool_schema(),
            self.resume_agent.resume_match_tool_schema(),
            self.resume_agent.resume_tweak_tool_schema(),
        ]

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

        tool_call = tool_calls[0]

        self.chat_history.append_message(
            AssistantMessage(
                role="assistant",
                content=choice.message.content or "",
                tool_calls=[tool_call.model_dump()] if tool_calls else None,
            )
        )

        args = json.loads(tool_call.function.arguments)
        tool_name = tool_call.function.name

        if tool_name == "search_jobs":
            query_args = QueryArgs(**args)
            result = self.query_agent.query_vectorstore(query_args)
        elif tool_name == "match_job_to_resume":
            resume_match_args = ResumeMatchArgs(**args)
            result = self.resume_agent.run_resume_match(resume_match_args)
        elif tool_name == "suggest_resume_tweaks":
            resume_tweak_args = ResumeTweakArgs(**args)
            result = self.resume_agent.run_resume_tweak(resume_tweak_args)
        else:
            self.log("OpenAI returned an invalid tool call")
            raise InvalidToolError()

        self.chat_history.append_message(
            ToolMessage(
                role="tool",
                content=result.model_dump_json(),
                name=tool_name,
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

        return reply, result
