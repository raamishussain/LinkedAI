import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "Missing OpenAI API key, make sure it's set in .env"

EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_COLLECTION = "jobsDB/"
DEFAULT_MODEL = "gpt-4o-mini"

CHAT_AGENT_SYSTEM_PROMPT = """
You are a helpful chatbot which helps a user find and apply for relevant jobs.

Your goal is to help the user with any job related questions, resume questions, and also
finding relevant job postings from LinkedIn. 

You have access to a QueryAgent which can query a database of vectorized job descriptions. Use
the necessary tool call to query the database if the user asks about finding jobs which fit a certain
description.
"""
