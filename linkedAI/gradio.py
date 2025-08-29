import os
import sys
import logging
from typing import Optional, Generator
import traceback
from datetime import datetime

import gradio as gr
from linkedAI.agents.chat_agent import ChatAgent
from linkedAI.agents.data_models import (
    ChatHistory,
    UserMessage,
    AssistantMessage,
)
from linkedAI.config import OPENAI_API_KEY, CHROMA_COLLECTION, RESUME_PATH


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("linkedai_gradio.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class LinkedAIGradioApp:
    """Gradio application for LinkedAI chat agent."""

    def __init__(self, max_iterations: int = 3):
        """
        Initialize the Gradio application.

        Args:
            max_iterations: Maximum iterations for chat agent
        """
        self.max_iterations = max_iterations
        self.agent: Optional[ChatAgent] = None
        self.chat_history = ChatHistory()
        self._initialize_agent()

    def _validate_environment(self) -> bool:
        """Validate required environment variables and files."""
        issues = []

        # Check OpenAI API key
        if not OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY not found in environment")

        # Check ChromaDB path
        if not os.path.exists(CHROMA_COLLECTION):
            issues.append(f"ChromaDB path not found: {CHROMA_COLLECTION}")

        # Check resume file
        if not os.path.exists(RESUME_PATH):
            issues.append(f"Resume file not found: {RESUME_PATH}")

        if issues:
            for issue in issues:
                logger.error(f"Environment validation failed: {issue}")
            return False

        logger.info("Environment validation passed")
        return True

    def _initialize_agent(self) -> None:
        """Initialize the chat agent with error handling."""
        if not self._validate_environment():
            raise RuntimeError(
                "Environment validation failed. Check logs for details."
            )

        self.agent = ChatAgent(max_iterations=self.max_iterations)
        logger.info("ChatAgent initialized successfully")

    def get_health_status(self) -> tuple[bool, str]:
        """Check application health status."""
        if not self._validate_environment():
            return False, "Environment validation failed"

        return True, "Application healthy"

    def chat_response(self, message: str) -> Generator[list[dict], None, None]:
        """
        Process user messages and stream responses.

        Args:
            message: Current user message

        Yields:
            Updated chat history with streaming response
        """
        if not message.strip():
            return

        # Add user message to history
        self.chat_history.append_message(
            UserMessage(role="user", content=message)
        )

        try:
            # Add empty assistant message for streaming updates
            self.chat_history.append_message(
                AssistantMessage(role="assistant", content="")
            )

            # Stream response from chat agent
            response_text = ""
            for chunk in self.agent.chat(message):
                response_text += chunk

                # Update the last message in chat history
                # with accumulated response
                self.chat_history.messages[-1].content = response_text
                yield self.chat_history.to_messages()

        except Exception as e:
            error_msg = f"**Error**: {str(e)}"
            logger.error(f"Chat error: {str(e)}")
            logger.error(traceback.format_exc())

            # If assistant message wasn't added yet, add it now with error
            if self.chat_history.messages[-1].role == "user":
                self.chat_history.append_message(
                    AssistantMessage(role="assistant", content=error_msg)
                )
            else:
                self.chat_history.messages[-1].content = error_msg
            yield self.chat_history.to_messages()

    def reset_conversation(self) -> tuple[list, str]:
        """Reset the agent's conversation history."""
        try:
            # Reinitialize agent to reset conversation
            self._initialize_agent()
            # Reset chat history to initial state
            self.chat_history = ChatHistory()
            logger.info("Conversation reset successfully")
            return (
                [],
                "CONVERSATION RESET -"
                " You can ask me anything about finding jobs or optimizing your resume.",  # noqa: E501
            )

        except Exception as e:
            logger.error(f"Reset conversation error: {str(e)}")
            return [], f"Error resetting conversation**: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface."""

        # Custom CSS for dark theme
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        .header-text {
            text-align: center;
            color: #e0e0e0;
        }
        .status-indicator {
            padding: 6px 12px;
            border-radius: 4px;
            margin: 5px 0;
            text-align: center;
            font-size: 0.9em;
        }
        .status-healthy {
            background-color: #1a3d2e;
            border: 1px solid #2d5a45;
            color: #4caf50;
        }
        .status-error {
            background-color: #4a1a1a;
            border: 1px solid #6d2c2c;
            color: #ff6b6b;
        }
        .compact-footer {
            font-size: 0.85em;
            color: #999;
            margin-top: 10px;
        }
        """

        with gr.Blocks(
            title="LinkedAI Job Search Assistant",
            theme=gr.themes.Soft().set(
                body_background_fill="#1a1a1a",
                body_text_color="#e0e0e0",
                background_fill_primary="#2a2a2a",
                background_fill_secondary="#333333",
            ),
            css=custom_css,
            analytics_enabled=False,
        ) as demo:

            # Header section
            with gr.Column(elem_classes="header-text"):
                gr.Markdown(
                    """
                # LinkedAI Job Search Assistant
                **AI-Powered Job Matching and Resume Optimization**
                """
                )

            # System status indicator
            health_status, health_message = self.get_health_status()
            status_class = (
                "status-healthy" if health_status else "status-error"
            )
            status_icon = "✅" if health_status else "❌"

            gr.Markdown(
                f"""
                <div class="status-indicator {status_class}">
                    {status_icon} <strong>System Status:</strong> {health_message}.
                </div>
                """,
                elem_classes="status-indicator",
            )

            # Usage instructions
            gr.Markdown(
                """
            ### 💡 What I can help you with:

            - **🔍 Job Search**: "Find data scientist jobs in San Francisco focusing on machine learning"
            - **📊 Resume Analysis**: "Which of these jobs best matches my resume?"
            - **✨ Resume Optimization**: "How can I improve my resume for this role?"
            - **🎯 Follow-up Questions**: Ask detailed questions about specific jobs or requirements
            """
            )

            chatbot = gr.Chatbot(
                height=700,
                show_copy_button=True,
                render_markdown=True,
                show_share_button=False,
                type="messages",
            )

            # Input section
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me to find jobs or help optimize your resume...",  # noqa: E501
                    container=False,
                    scale=7,
                    max_lines=3,
                )
                submit_btn = gr.Button("Send", scale=1, variant="primary")

            # Control buttons
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", scale=1)
                reset_btn = gr.Button(
                    "Reset Conversation", scale=1, variant="secondary"
                )
                status_btn = gr.Button("Check Status", scale=1, variant="stop")

            # Event handlers
            msg.submit(
                self.chat_response,
                inputs=[msg],
                outputs=[chatbot],
                show_progress=True,
                concurrency_limit=5,
            ).then(lambda: gr.update(value=""), outputs=[msg])

            submit_btn.click(
                self.chat_response,
                inputs=[msg],
                outputs=[chatbot],
                show_progress=True,
                concurrency_limit=5,
            ).then(lambda: gr.update(value=""), outputs=[msg])

            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

            reset_btn.click(self.reset_conversation, outputs=[chatbot, msg])

            def update_status():
                _, health_message = self.get_health_status()
                timestamp = datetime.now().strftime("%H:%M:%S")
                return f"**Status Check ({timestamp})**: {health_message}"

            status_btn.click(update_status, outputs=[msg])

            # Footer
            gr.Markdown(
                """
            ---
            ### ⚠️ Important Notes:
            - **Data Privacy**: Your conversations are not stored permanently
            - **API Usage**: This application uses OpenAI's API for processing
            - **Resume Privacy**: Your resume is processed locally and not shared externally  
            - **Job Data**: Job information is sourced from your local ChromaDB
            """
            )

        return demo


def create_app(
    max_iterations: int = 3,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> gr.Blocks:
    """
    Factory function to create and configure the Gradio app.

    Args:
        max_iterations: Maximum chat agent iterations
        server_name: Server host address
        server_port: Server port
        share: Whether to create a public shareable link

    Returns:
        Configured Gradio Blocks interface
    """
    app = LinkedAIGradioApp(max_iterations=max_iterations)
    demo = app.create_interface()

    return demo, {
        "server_name": server_name,
        "server_port": server_port,
        "share": share,
        "show_error": True,
        "quiet": False,
    }


def main():
    """Main entry point for the application."""

    try:
        demo, launch_config = create_app(
            max_iterations=3,
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
        )

        logger.info("Starting Gradio server...")
        demo.launch(**launch_config)

    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
