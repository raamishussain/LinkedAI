import os
import sys
import logging
from typing import Optional, Generator
import traceback
from datetime import datetime

import gradio as gr
from linkedAI.agents.chat_agent import ChatAgent
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
        try:
            if not self._validate_environment():
                raise RuntimeError(
                    "Environment validation failed. Check logs for details."
                )

            self.agent = ChatAgent(max_iterations=self.max_iterations)
            logger.info("ChatAgent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize ChatAgent: {str(e)}")
            logger.error(traceback.format_exc())
            self.agent = None

    def get_health_status(self) -> tuple[bool, str]:
        """Check application health status."""
        if not self.agent:
            return False, "ChatAgent not initialized"

        if not self._validate_environment():
            return False, "Environment validation failed"

        return True, "Application healthy"

    def chat_response(
        self, message: str, history: list[list[str]]
    ) -> Generator[list[list[str]], None, None]:
        """
        Process user messages and stream responses.

        Args:
            message: Current user message
            history: Chat history managed by Gradio

        Yields:
            Updated chat history with streaming response
        """
        if not message.strip():
            return

        if not self.agent:
            error_msg = "**System Error**: Chat agent not available."
            history.append([message, error_msg])
            yield history
            return

        # Add user message to history
        history.append([message, ""])

        try:
            # Stream response from chat agent
            response_text = ""
            for chunk in self.agent.chat(message):
                response_text += chunk
                # Update the last message in history with accumulated response
                history[-1][1] = response_text
                yield history

        except Exception as e:
            error_msg = f"**Error**: {str(e)}"
            logger.error(f"Chat error: {str(e)}")
            logger.error(traceback.format_exc())

            history[-1][1] = error_msg
            yield history

    def reset_conversation(self) -> tuple[list, str]:
        """Reset the agent's conversation history."""
        try:
            if self.agent:
                # Reinitialize agent to reset conversation
                self._initialize_agent()
                logger.info("Conversation reset successfully")
                return (
                    [],
                    "CONVERSATION RESET -"
                    " You can ask me anything about finding jobs or optimizing your resume.",  # noqa: E501
                )
            else:
                return ([], "Unable to reset - agent not available.")

        except Exception as e:
            logger.error(f"Reset conversation error: {str(e)}")
            return [], f"Error resetting conversation**: {str(e)}"

    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface."""

        # Custom CSS for better production appearance
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .header-text {
            text-align: center;
            color: #2c3e50;
        }
        .status-indicator {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            text-align: center;
        }
        .status-healthy {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .status-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        """

        with gr.Blocks(
            title="LinkedAI Job Search Assistant",
            theme=gr.themes.Soft(),
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
                height=500,
                show_copy_button=True,
                bubble_full_width=False,
                render_markdown=True,
                show_share_button=False,
                avatar_images=("🧑‍💼", "🤖"),
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
                inputs=[msg, chatbot],
                outputs=[chatbot],
                show_progress=True,
                concurrency_limit=5,
            ).then(lambda: gr.update(value=""), outputs=[msg])

            submit_btn.click(
                self.chat_response,
                inputs=[msg, chatbot],
                outputs=[chatbot],
                show_progress=True,
                concurrency_limit=5,
            ).then(lambda: gr.update(value=""), outputs=[msg])

            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg])

            reset_btn.click(self.reset_conversation, outputs=[chatbot, msg])

            def update_status():
                health_status, health_message = self.get_health_status()
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
    enable_auth: bool = False,
    auth_username: Optional[str] = None,
    auth_password: Optional[str] = None,
) -> gr.Blocks:
    """
    Factory function to create and configure the Gradio app.

    Args:
        max_iterations: Maximum chat agent iterations
        server_name: Server host address
        server_port: Server port
        share: Whether to create a public shareable link
        enable_auth: Whether to enable basic authentication
        auth_username: Username for authentication
        auth_password: Password for authentication

    Returns:
        Configured Gradio Blocks interface
    """
    app = LinkedAIGradioApp(max_iterations=max_iterations)
    demo = app.create_interface()

    # Configure authentication if enabled
    auth = None
    if enable_auth and auth_username and auth_password:
        auth = (auth_username, auth_password)
        logger.info("Authentication enabled")

    return demo, {
        "server_name": server_name,
        "server_port": server_port,
        "share": share,
        "auth": auth,
        "show_error": True,
        "quiet": False,
    }


def main():
    """Main entry point for the application."""

    # Configuration from environment variables with defaults
    MAX_ITERATIONS = int(os.getenv("LINKEDAI_MAX_ITERATIONS", "3"))
    SERVER_NAME = os.getenv("LINKEDAI_SERVER_NAME", "127.0.0.1")
    SERVER_PORT = int(os.getenv("LINKEDAI_SERVER_PORT", "7860"))
    SHARE = os.getenv("LINKEDAI_SHARE", "false").lower() == "true"

    # Authentication settings
    ENABLE_AUTH = os.getenv("LINKEDAI_ENABLE_AUTH", "false").lower() == "true"
    AUTH_USERNAME = os.getenv("LINKEDAI_AUTH_USERNAME")
    AUTH_PASSWORD = os.getenv("LINKEDAI_AUTH_PASSWORD")

    print("🚀 Starting LinkedAI Chat Interface...")
    print(f"📍 Server: {SERVER_NAME}:{SERVER_PORT}")
    print(f"🔄 Max iterations: {MAX_ITERATIONS}")
    print(f"🔗 Share: {SHARE}")
    print(f"🔐 Authentication: {ENABLE_AUTH}")

    print("\n📋 Prerequisites:")
    print("1. ✅ OPENAI_API_KEY set in environment")
    print("2. ✅ ChromaDB with job data available")
    print("3. ✅ Resume file at configured path")
    print()

    try:
        demo, launch_config = create_app(
            max_iterations=MAX_ITERATIONS,
            server_name=SERVER_NAME,
            server_port=SERVER_PORT,
            share=SHARE,
            enable_auth=ENABLE_AUTH,
            auth_username=AUTH_USERNAME,
            auth_password=AUTH_PASSWORD,
        )

        logger.info("Starting Gradio server...")
        demo.launch(**launch_config)

    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
        print("\n👋 Server stopped gracefully")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\n❌ Server failed to start: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
