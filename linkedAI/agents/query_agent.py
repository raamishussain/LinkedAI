from chromadb.api.models.Collection import Collection
from linkedAI.agents import Agent, QueryArgs, SearchResults
from linkedAI.config import EMBEDDING_MODEL, OPENAI_API_KEY
from linkedAI.scraper.data_models import JobCard
from openai import OpenAI
from typing import Any


class QueryAgent(Agent):
    """Agent class to handle queries to the vector database"""

    name = "QueryAgent"

    def __init__(self, collection: Collection):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.collection = collection
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
            where=query_args.filters or {},
        )

        jobs = []
        for i in range(len(results["documents"])):
            jobs.append(
                JobCard(
                    description=results["documents"][0][i],
                    **results["metadatas"][0][i]
                )
            )

        return jobs

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
