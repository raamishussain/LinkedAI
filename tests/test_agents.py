import pytest
import json
from unittest.mock import Mock, patch
from pathlib import Path

from linkedAI.agents.agent import Agent
from linkedAI.agents.chat_agent import (
    ChatAgent,
    ChatSession,
    InvalidToolError,
)
from linkedAI.agents.query_agent import QueryAgent
from linkedAI.agents.resume_agent import ResumeAgent
from linkedAI.agents.data_models import (
    ChatHistory,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    QueryArgs,
    SearchResults,
    ResumeMatchArgs,
    ResumeMatchResult,
    ResumeTweakArgs,
    ResumeTweakResult,
)
from pydantic import ValidationError


class TestAgent:
    """Test the base Agent class"""

    def test_agent_initialization(self):
        agent = Agent("TestAgent")
        assert agent.name == "TestAgent"
        assert agent.logger.name == "TestAgent"


class TestChatHistory:
    """Test ChatHistory functionality"""

    def test_chat_history_initialization(self):
        history = ChatHistory()
        assert len(history.messages) == 1
        assert isinstance(history.messages[0], SystemMessage)

    def test_append_message(self):
        history = ChatHistory()
        user_msg = UserMessage(role="user", content="Hello")
        history.append_message(user_msg)
        assert len(history.messages) == 2
        assert history.messages[1] == user_msg

    def test_to_messages_format(self):
        history = ChatHistory()
        user_msg = UserMessage(role="user", content="Hello")
        assistant_msg = AssistantMessage(role="assistant", content="Hi there")

        history.append_message(user_msg)
        history.append_message(assistant_msg)

        messages = history.to_messages()
        assert len(messages) == 3  # system + user + assistant
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi there"

    def test_tool_message_formatting(self):
        history = ChatHistory()
        tool_msg = ToolMessage(
            role="tool",
            content="Tool result",
            tool_call_id="call_123",
            name="test_tool",
        )
        history.append_message(tool_msg)

        messages = history.to_messages()
        tool_message = messages[1]
        assert tool_message["role"] == "tool"
        assert tool_message["content"] == "Tool result"
        assert tool_message["tool_call_id"] == "call_123"
        assert tool_message["name"] == "test_tool"


class TestChatSession:
    """Test ChatSession functionality"""

    def test_chat_session_initialization(self):
        session = ChatSession()
        assert session.iterations == 0
        assert session.max_iterations == 5
        assert session.tool_results == {}
        assert session.last_search_results is None


class TestQueryAgent:
    """Test QueryAgent functionality"""

    @patch("chromadb.PersistentClient")
    def test_query_agent_initialization(
        self, mock_chroma_client, mock_openai_client
    ):
        mock_collection = Mock()
        mock_chroma_instance = Mock()
        mock_chroma_instance.get_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_chroma_instance

        agent = QueryAgent(
            openai_client=mock_openai_client,
            chroma_path="/test/path",
            collection="test_collection",
        )

        assert agent.client == mock_openai_client
        assert agent.collection == mock_collection
        mock_chroma_client.assert_called_once_with(path="/test/path")
        mock_chroma_instance.get_collection.assert_called_once_with(
            "test_collection"
        )

    def test_query_vectorstore(
        self, mock_openai_client, mock_chroma_collection
    ):
        with patch("chromadb.PersistentClient") as mock_chroma_client:
            mock_chroma_instance = Mock()
            mock_chroma_instance.get_collection.return_value = (
                mock_chroma_collection
            )
            mock_chroma_client.return_value = mock_chroma_instance

            agent = QueryAgent(
                openai_client=mock_openai_client,
                chroma_path="/test/path",
                collection="test_collection",
            )

            query_args = QueryArgs(query="python developer", n_results=5)
            result = agent.query_vectorstore(query_args)

            assert isinstance(result, SearchResults)
            assert len(result.jobs) == 1
            assert result.jobs[0].title == "Test Job"

            mock_openai_client.embeddings.create.assert_called_once()
            mock_chroma_collection.query.assert_called_once()

    def test_tool_schema(self):
        schema = QueryAgent.tool_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_jobs"
        assert "parameters" in schema["function"]


class TestResumeAgent:
    """Test ResumeAgent functionality"""

    def test_resume_agent_initialization_missing_file(
        self, mock_openai_client
    ):
        non_existent_path = Path("/non/existent/path.txt")
        agent = ResumeAgent(
            openai_client=mock_openai_client, resume_path=non_existent_path
        )
        assert agent.resume == ""

    def test_load_resume_pdf(
        self,
        datadir,
        mock_openai_client,
    ):
        resume_path = datadir / "test_resume.pdf"
        agent = ResumeAgent(
            openai_client=mock_openai_client, resume_path=resume_path
        )

        assert agent.resume != ""

    def test_run_resume_match(
        self, datadir, mock_openai_client, sample_search_results
    ):
        resume_path = datadir / "test_resume.pdf"
        agent = ResumeAgent(
            openai_client=mock_openai_client, resume_path=resume_path
        )

        match_args = ResumeMatchArgs(jobs=sample_search_results)
        result = agent.run_resume_match(match_args)

        assert isinstance(result, ResumeMatchResult)
        assert result.best_match_id == 0
        assert result.reasoning == "Test reasoning"

        mock_openai_client.beta.chat.completions.parse.assert_called_once()

    def test_run_resume_match_null_response(
        self, datadir, mock_openai_client, sample_search_results
    ):
        # Mock null response
        structured_response = Mock()
        structured_choice = Mock()
        structured_choice.message.parsed = None
        structured_response.choices = [structured_choice]
        mock_openai_client.beta.chat.completions.parse.return_value = (
            structured_response
        )

        resume_path = datadir / "test_resume.pdf"
        agent = ResumeAgent(
            openai_client=mock_openai_client, resume_path=resume_path
        )

        match_args = ResumeMatchArgs(jobs=sample_search_results)
        result = agent.run_resume_match(match_args)

        assert result.best_match_id == 0
        assert result.reasoning == ""

    def test_run_resume_tweak(self, datadir, mock_openai_client):
        resume_path = datadir / "test_resume.pdf"
        agent = ResumeAgent(
            openai_client=mock_openai_client, resume_path=resume_path
        )

        tweak_args = ResumeTweakArgs(job_description="Python developer role")
        result = agent.run_resume_tweak(tweak_args)

        assert isinstance(result, ResumeTweakResult)
        assert result.suggestions == "Test response"

        mock_openai_client.chat.completions.create.assert_called_once()

    def test_resume_match_tool_schema(self):
        schema = ResumeAgent.resume_match_tool_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "match_job_to_resume"

    def test_resume_tweak_tool_schema(self):
        schema = ResumeAgent.resume_tweak_tool_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "suggest_resume_tweaks"


class TestChatAgent:
    """Test ChatAgent functionality"""

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_chat_agent_initialization(
        self, mock_resume_agent, mock_query_agent, mock_openai_client
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):
            agent = ChatAgent(max_iterations=5)

            assert agent.max_iterations == 5
            assert agent.client == mock_openai_client
            assert isinstance(agent.chat_history, ChatHistory)

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_tools_method(
        self, mock_resume_agent, mock_query_agent, mock_openai_client
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):
            agent = ChatAgent()
            tools = agent._tools()

            assert len(tools) == 3
            mock_query_agent.return_value.tool_schema.assert_called_once()
            mock_resume_agent.return_value.resume_match_tool_schema.assert_called_once()  # noqa: E501
            mock_resume_agent.return_value.resume_tweak_tool_schema.assert_called_once()  # noqa: E501

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_execute_tool_call_search_jobs(
        self,
        mock_resume_agent,
        mock_query_agent,
        mock_openai_client,
        sample_search_results,
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):
            agent = ChatAgent()
            mock_query_agent.return_value.query_vectorstore.return_value = (
                sample_search_results
            )

            tool_call = Mock()
            tool_call.function = Mock()
            tool_call.function.name = "search_jobs"
            tool_call.function.arguments = json.dumps(
                {"query": "python", "n_results": 5}
            )
            tool_call.id = "call_123"

            session = ChatSession()
            result = agent._execute_tool_call(tool_call, session)

            assert result == sample_search_results
            assert session.last_search_results == sample_search_results
            mock_query_agent.return_value.query_vectorstore.assert_called_once()  # noqa: E501

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_execute_tool_call_match_job_to_resume(
        self,
        mock_resume_agent,
        mock_query_agent,
        mock_openai_client,
        sample_search_results,
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):
            agent = ChatAgent()
            mock_result = ResumeMatchResult(
                best_match_id=1, reasoning="Good match"
            )
            mock_resume_agent.return_value.run_resume_match.return_value = (
                mock_result
            )

            tool_call = Mock()
            tool_call.function = Mock()
            tool_call.function.name = "match_job_to_resume"
            tool_call.function.arguments = json.dumps(
                {"jobs": sample_search_results.model_dump()}
            )
            tool_call.id = "call_123"

            session = ChatSession()
            session.last_search_results = sample_search_results
            result = agent._execute_tool_call(tool_call, session)

            assert result == mock_result
            mock_resume_agent.return_value.run_resume_match.assert_called_once()  # noqa: E501

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_execute_tool_call_suggest_resume_tweaks(
        self, mock_resume_agent, mock_query_agent, mock_openai_client
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):
            agent = ChatAgent()
            mock_result = ResumeTweakResult(
                suggestions="Add more Python experience"
            )
            mock_resume_agent.return_value.run_resume_tweak.return_value = (
                mock_result
            )

            tool_call = Mock()
            tool_call.function = Mock()
            tool_call.function.name = "suggest_resume_tweaks"
            tool_call.function.arguments = json.dumps(
                {"job_description": "Python developer role"}
            )
            tool_call.id = "call_123"

            session = ChatSession()
            result = agent._execute_tool_call(tool_call, session)

            assert result == mock_result
            mock_resume_agent.return_value.run_resume_tweak.assert_called_once()  # noqa: E501

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_execute_tool_call_invalid_tool(
        self, mock_resume_agent, mock_query_agent, mock_openai_client
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):
            agent = ChatAgent()

            tool_call = Mock()
            tool_call.function = Mock()
            tool_call.function.name = "invalid_tool"
            tool_call.function.arguments = "{}"
            tool_call.id = "call_123"

            session = ChatSession()

            with pytest.raises(InvalidToolError):
                agent._execute_tool_call(tool_call, session)

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_execute_tool_call_missing_jobs_for_resume_match(
        self, mock_resume_agent, mock_query_agent, mock_openai_client
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):
            agent = ChatAgent()

            tool_call = Mock()
            tool_call.function = Mock()
            tool_call.function.name = "match_job_to_resume"
            tool_call.function.arguments = json.dumps({})
            tool_call.id = "call_123"

            session = ChatSession()  # No last_search_results

            with pytest.raises(ValidationError):
                agent._execute_tool_call(tool_call, session)

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_process_all_tool_calls_success(
        self,
        mock_resume_agent,
        mock_query_agent,
        mock_openai_client,
        sample_search_results,
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):
            agent = ChatAgent()
            mock_query_agent.return_value.query_vectorstore.return_value = (
                sample_search_results
            )

            tool_call = Mock()
            tool_call.function.name = "search_jobs"
            tool_call.function.arguments = json.dumps(
                {"query": "python", "n_results": 5}
            )
            tool_call.id = "call_123"

            session = ChatSession()
            results = agent._process_all_tool_calls([tool_call], session)

            assert len(results) == 1
            assert results[0] == sample_search_results
            assert len(agent.chat_history.messages) > 1  # Tool message added

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_process_all_tool_calls_exception(
        self, mock_resume_agent, mock_query_agent, mock_openai_client
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):
            agent = ChatAgent()
            mock_query_agent.return_value.query_vectorstore.side_effect = (
                Exception("Test error")
            )

            tool_call = Mock()
            tool_call.function.name = "search_jobs"
            tool_call.function.arguments = json.dumps(
                {"query": "python", "n_results": 5}
            )
            tool_call.id = "call_123"

            session = ChatSession()
            results = agent._process_all_tool_calls([tool_call], session)

            assert len(results) == 1
            assert results[0] is None  # Error result

            # Check that error was logged to chat history
            tool_messages = [
                msg
                for msg in agent.chat_history.messages
                if isinstance(msg, ToolMessage)
            ]
            assert len(tool_messages) == 1
            assert "Error: Test error" in tool_messages[0].content

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_chat_no_tool_calls(
        self, mock_resume_agent, mock_query_agent, mock_openai_client
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):
            # Mock response with no tool calls
            choice = Mock()
            choice.message.content = "Hello! How can I help you?"
            choice.message.tool_calls = None
            response = Mock()
            response.choices = [choice]
            mock_openai_client.chat.completions.create.return_value = response

            agent = ChatAgent()
            responses = list(agent.chat("Hello"))

            assert len(responses) == 1
            assert responses[0] == "Hello! How can I help you?"

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_chat_with_tool_calls(
        self,
        mock_resume_agent,
        mock_query_agent,
        mock_openai_client,
        sample_search_results,
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):

            # Create mock tool call object
            mock_tool_call = Mock()
            mock_tool_call.function.name = "search_jobs"
            mock_tool_call.function.arguments = (
                '{"query": "python", "n_results": 5}'
            )
            mock_tool_call.id = "call_123"
            mock_tool_call.model_dump.return_value = {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "search_jobs",
                    "arguments": '{"query": "python", "n_results": 5}',
                },
            }

            choice1 = Mock()
            choice1.message.content = ""
            choice1.message.tool_calls = [mock_tool_call]
            response1 = Mock()
            response1.choices = [choice1]

            # Mock second response without tool calls (final response)
            choice2 = Mock()
            choice2.message.content = "Here are the search results!"
            choice2.message.tool_calls = None
            response2 = Mock()
            response2.choices = [choice2]

            mock_openai_client.chat.completions.create.side_effect = [
                response1,
                response2,
            ]
            mock_query_agent.return_value.query_vectorstore.return_value = (
                sample_search_results
            )

            agent = ChatAgent()
            responses = list(agent.chat("Find me Python jobs"))

            # Should get progress message and final response
            assert len(responses) >= 2
            assert "Searching for jobs..." in responses[0]
            assert "Here are the search results!" in responses[-1]

    @patch("linkedAI.agents.chat_agent.QueryAgent")
    @patch("linkedAI.agents.chat_agent.ResumeAgent")
    def test_chat_max_iterations_reached(
        self, mock_resume_agent, mock_query_agent, mock_openai_client
    ):
        with patch(
            "linkedAI.agents.chat_agent.OpenAI",
            return_value=mock_openai_client,
        ):

            # Create mock tool call object
            mock_tool_call_iter = Mock()
            mock_tool_call_iter.function.name = "search_jobs"
            mock_tool_call_iter.function.arguments = (
                '{"query": "python", "n_results": 5}'
            )
            mock_tool_call_iter.id = "call_123"
            mock_tool_call_iter.model_dump.return_value = {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "search_jobs",
                    "arguments": '{"query": "python", "n_results": 5}',
                },
            }

            choice_with_tools = Mock()
            choice_with_tools.message.content = ""
            choice_with_tools.message.tool_calls = [mock_tool_call_iter]
            response_with_tools = Mock()
            response_with_tools.choices = [choice_with_tools]

            # Final response without tools
            choice_final = Mock()
            choice_final.message.content = "Final summary"
            choice_final.message.tool_calls = None
            response_final = Mock()
            response_final.choices = [choice_final]

            # Return tool responses until max iterations, then final response
            mock_openai_client.chat.completions.create.side_effect = [
                response_with_tools,  # iteration 1
                response_with_tools,  # iteration 2
                response_with_tools,  # iteration 3 (max)
                response_final,  # final call
            ]

            agent = ChatAgent(max_iterations=3)
            responses = list(agent.chat("Find me jobs"))

            # Should include "Finalizing response..." and final content
            assert any("Finalizing response..." in r for r in responses)
            assert responses[-1] == "Final summary"
