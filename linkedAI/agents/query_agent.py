from chromadb.api.models.Collection import Collection
from linkedAI.agents.agent import Agent
from linkedAI.config import EMBEDDING_MODEL, OPENAI_API_KEY
from openai import OpenAI


class QueryAgent(Agent):
    """Agent class to handle queries to the vector database"""

    name = "QueryAgent"

    def __init__(self, collection: Collection):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.collection = collection
        self.embedding_model = EMBEDDING_MODEL
        self.log("Successfully initialized QueryAgent")

    def query_vectorstore(
        self,
        client: OpenAI,
        collection: Collection,
        query: str,
        n_results: int = 1,
    ) -> dict:
        """Query the vectorstore for relevant jobs

        Args:
            client: OpenAI
                OpenAI client
            collection: Collection
                ChromaDB collection where job vector embeddings are stored
            query: str
                user query for job
            n_results: int
                Number of jobs to retreive based on highest similarity

        Returns:
            results: dict
                dictionary of results with n_results number of jobs
        """

        response = client.embeddings.create(
            model=self.embedding_model,
            input=query,
        )

        results = collection.query(
            query_embeddings=response.data[0].embedding,
            n_results=n_results,
            include=["documents", "metadatas"],
        )

        return results
