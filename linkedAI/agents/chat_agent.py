import json

from dataclasses import dataclass
from linkedAI.agents.agent import Agent
from linkedAI.agents.data_models import (
    AssistantMessage,
    ChatHistory,
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
from typing import Any, Generator, Optional


class InvalidToolError(Exception):
    """Exception raised for invalid tool call"""

    pass


@dataclass
class ChatSession:
    """Tracks state during multi-tool conversations"""

    iterations: int = 0
    max_iterations: int = 5
    tool_results: dict[str, Any] = None
    last_search_results: Optional[SearchResults] = None

    def __post_init__(self):
        if self.tool_results is None:
            self.tool_results = {}


class ChatAgent(Agent):
    """User facing agent class which facilitates basic chat and can call to
    other agents when needed"""

    def __init__(self, max_iterations: int = 3):

        super().__init__("ChatAgent")

        self.model = DEFAULT_MODEL
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.max_iterations = max_iterations

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

    def _execute_tool_call(self, tool_call, session: ChatSession) -> Any:
        """Execute a single tool call and return the result"""

        args = json.loads(tool_call.function.arguments)
        tool_name = tool_call.function.name

        self.log(f"Executing Tool: {tool_name}")

        if tool_name == "search_jobs":
            query_args = QueryArgs(**args)
            result = self.query_agent.query_vectorstore(query_args)
            session.last_search_results = result
            session.tool_results["last_search"] = result
            return result

        elif tool_name == "match_job_to_resume":
            resume_match_args = ResumeMatchArgs(**args)

            # if no jobs are provided, use previous search results
            if (
                not hasattr(resume_match_args, "jobs")
                or not resume_match_args.jobs
            ):
                if session.last_search_results:
                    resume_match_args.jobs = session.last_search_results
                else:
                    raise InvalidToolError(
                        "No jobs provided for resume matching"
                    )
            result = self.resume_agent.run_resume_match(resume_match_args)
            session.tool_results["last_match"] = result
            return result

        elif tool_name == "suggest_resume_tweaks":
            resume_tweak_args = ResumeTweakArgs(**args)
            result = self.resume_agent.run_resume_tweak(resume_tweak_args)
            session.tool_results["last_tweaks"] = result
            return result
        else:
            self.log("OpenAI returned an invalid tool call", level="error")
            raise InvalidToolError(f"Unknown tool: {tool_name}")

    def _process_all_tool_calls(
        self, tool_calls, session: ChatSession
    ) -> list[Any]:
        """Process all tool calls from a single LLM response"""
        results = []

        for tool_call in tool_calls:
            try:
                result = self._execute_tool_call(tool_call, session)
                results.append(result)

                self.chat_history.append_message(
                    ToolMessage(
                        role="tool",
                        content=result.model_dump_json(),
                        name=tool_call.function.name,
                        tool_call_id=tool_call.id,
                    )
                )
            except Exception as e:
                self.log(f"Tool execution failed: {e}", level="error")
                self.chat_history.append_message(
                    ToolMessage(
                        role="tool",
                        content=f"Error: {str(e)}",
                        name=tool_call.function.name,
                        tool_call_id=tool_call.id,
                    )
                )
                results.append(None)

        return results

    def _format_job_results(self, search_results: SearchResults) -> str:
        """Format job search results for display in chat interface"""
        if not search_results or not search_results.jobs:
            return "No jobs found matching your criteria."

        formatted = f"Found {len(search_results.jobs)} relevant jobs:\n\n"

        for i, job in enumerate(search_results.jobs, 1):
            formatted += f"**{i}. {job.title}** at **{job.company}**\n"
            formatted += f"{job.location}\n"
            formatted += f"[Apply Here]({job.link})\n"

            # Truncate description to first 200 chars for readability
            description = job.description
            if len(description) > 200:
                description = description[:200] + "..."
            formatted += f"{description}\n\n"

        return formatted

    def _format_resume_match_result(self, match_result: Any) -> str:
        """Format resume match result for display"""
        if not match_result:
            return "Unable to analyze resume match."

        formatted = "**Best Match Analysis:**\n\n"
        formatted += (
            f"**Best Job Match:** Job #{match_result.best_match_id}\n\n"
        )
        formatted += f"**Reasoning:**\n{match_result.reasoning}\n\n"

        return formatted

    def _format_resume_tweaks(self, tweak_result: Any) -> str:
        """Format resume tweak suggestions for display"""
        if not tweak_result:
            return "No resume suggestions available."

        formatted = "**Resume Optimization Suggestions:**\n\n"
        formatted += f"{tweak_result.suggestions}\n\n"

        return formatted

    def chat(self, user_text: str) -> Generator[str, None, None]:
        """Perform chat between user and LLM with streamed responses

        Args:
            query: str
                User query to send to the model

        Yields:
            str: Incremental text responses showing progress and final results
        """
        session = ChatSession(max_iterations=self.max_iterations)

        # Add user message to history
        self.chat_history.append_message(
            UserMessage(role="user", content=user_text)
        )

        yield "Let me help you with that...\n\n"

        while session.iterations < session.max_iterations:
            session.iterations += 1

            # Get LLM response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.chat_history.to_messages(),
                tools=self._tools(),
            )

            choice = response.choices[0]
            tool_calls = choice.message.tool_calls

            # Add assistant message to history
            self.chat_history.append_message(
                AssistantMessage(
                    role="assistant",
                    content=choice.message.content or "",
                    tool_calls=[tc.model_dump() for tc in tool_calls]
                    if tool_calls
                    else None,
                )
            )

            # If no tool calls, we're done - yield final response
            if not tool_calls:
                if choice.message.content:
                    yield choice.message.content
                return

            # Process tool calls and stream results
            for tool_call in tool_calls:
                tool_name = tool_call.function.name

                if tool_name == "search_jobs":
                    yield "Searching for jobs...\n"
                    try:
                        args = json.loads(tool_call.function.arguments)
                        query_args = QueryArgs(**args)
                        result = self.query_agent.query_vectorstore(query_args)

                        # Store for potential use by other tools
                        session.last_search_results = result
                        session.tool_results["last_search"] = result

                        # Add tool result to conversation history
                        self.chat_history.append_message(
                            ToolMessage(
                                role="tool",
                                content=result.model_dump_json(),
                                name=tool_name,
                                tool_call_id=tool_call.id,
                            )
                        )

                        # Stream formatted results
                        formatted_jobs = self._format_job_results(result)
                        yield formatted_jobs

                    except Exception as e:
                        error_msg = f"Error searching jobs: {str(e)}\n"
                        yield error_msg
                        self.chat_history.append_message(
                            ToolMessage(
                                role="tool",
                                content=f"Error: {str(e)}",
                                name=tool_name,
                                tool_call_id=tool_call.id,
                            )
                        )

                elif tool_name == "match_job_to_resume":
                    yield "Analyzing your resume against the jobs...\n"
                    try:
                        args = json.loads(tool_call.function.arguments)
                        resume_match_args = ResumeMatchArgs(**args)

                        # Use last search results if jobs not provided
                        if (
                            not hasattr(resume_match_args, "jobs")
                            or not resume_match_args.jobs
                        ):
                            if session.last_search_results:
                                resume_match_args.jobs = (
                                    session.last_search_results
                                )
                            else:
                                yield "No jobs available for resume matching\n"
                                continue

                        result = self.resume_agent.run_resume_match(
                            resume_match_args
                        )
                        session.tool_results["last_match"] = result

                        # Add tool result to conversation history
                        self.chat_history.append_message(
                            ToolMessage(
                                role="tool",
                                content=result.model_dump_json(),
                                name=tool_name,
                                tool_call_id=tool_call.id,
                            )
                        )

                        # Stream formatted results
                        formatted_match = self._format_resume_match_result(
                            result
                        )
                        yield formatted_match

                    except Exception as e:
                        error_msg = f"Error analyzing resume: {str(e)}\n"
                        yield error_msg
                        self.chat_history.append_message(
                            ToolMessage(
                                role="tool",
                                content=f"Error: {str(e)}",
                                name=tool_name,
                                tool_call_id=tool_call.id,
                            )
                        )

                elif tool_name == "suggest_resume_tweaks":
                    yield "Generating resume optimization suggestions...\n"
                    try:
                        args = json.loads(tool_call.function.arguments)
                        resume_tweak_args = ResumeTweakArgs(**args)
                        result = self.resume_agent.run_resume_tweak(
                            resume_tweak_args
                        )
                        session.tool_results["last_tweaks"] = result

                        # Add tool result to conversation history
                        self.chat_history.append_message(
                            ToolMessage(
                                role="tool",
                                content=result.model_dump_json(),
                                name=tool_name,
                                tool_call_id=tool_call.id,
                            )
                        )

                        # Stream formatted results
                        formatted_tweaks = self._format_resume_tweaks(result)
                        yield formatted_tweaks

                    except Exception as e:
                        error_msg = f"Error generating suggestions: {str(e)}\n"
                        yield error_msg
                        self.chat_history.append_message(
                            ToolMessage(
                                role="tool",
                                content=f"Error: {str(e)}",
                                name=tool_name,
                                tool_call_id=tool_call.id,
                            )
                        )

        # If we hit max iterations, make final call for summary
        if session.iterations >= session.max_iterations:
            yield "Finalizing response...\n"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.chat_history.to_messages(),
                # No tools - force a text response
            )

            final_content = response.choices[0].message.content
            if final_content:
                # Add final response to history
                self.chat_history.append_message(
                    AssistantMessage(role="assistant", content=final_content)
                )
                yield final_content
