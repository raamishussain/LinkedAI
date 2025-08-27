import chromadb

from linkedAI.agents.agent import Agent
from linkedAI.agents.data_models import QueryArgs, SearchResults
from linkedAI.config import EMBEDDING_MODEL
from linkedAI.scraper.data_models import JobCard
from openai import OpenAI
from typing import Any


class QueryAgent(Agent):
    """Agent class to handle queries to the vector database"""

    name = "QueryAgent"

    def __init__(
        self, openai_client: OpenAI, chroma_path: str, collection: str
    ):
        self.client = openai_client
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = chroma_client.get_collection(collection)
        self.embedding_model = EMBEDDING_MODEL
        self.log("Successfully initialized QueryAgent")

    def query_vectorstore(self, query_args: QueryArgs) -> SearchResults:
        """Query the vectorstore for relevant jobs

        Args:
            query_args: QueryArgs
                arguments needed to query the vector DB

        Returns:
            results: SearchResults
                List of Job objects returned by the search
        """

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query_args.query,
        )

        results = self.collection.query(
            query_embeddings=response.data[0].embedding,
            n_results=query_args.n_results,
            include=["documents", "metadatas"],
        )

        jobs = []
        for i in range(len(results["documents"])):
            jobs.append(
                JobCard(
                    description=results["documents"][0][i],
                    **results["metadatas"][0][i]
                )
            )

        return SearchResults(jobs=jobs)

    @staticmethod
    def tool_schema() -> dict[str, Any]:
        """Return tool schema for OpenAI function calling."""
        return {
            "type": "function",
            "function": {
                "name": "search_jobs",
                "description": "Query the Jobs database for jobs matching the user's query",  # noqa: E501
                "parameters": QueryArgs.model_json_schema(),
            },
        }
