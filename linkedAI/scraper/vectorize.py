import argparse
import chromadb
import json
import logging
import sys
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def init_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def parse_job_json(json_path: str) -> tuple[dict[str], list[str], list[str]]:
    with open(json_path, "r") as f:
        jobs_json = json.load(f)

    metadatas = []
    ids = []
    descriptions = []

    for job in jobs_json:
        metadata = {
            "title": job["title"],
            "link": job["link"],
            "location": job["location"],
            "company": job["company"],
        }
        metadatas.append(metadata)

        # get job id from end of job link
        ids.append(
            job["link"].split("-")[-1]
        )

        descriptions.append(job["description"])

    return metadatas, ids, descriptions


def vectorize(
    metadatas: dict[str],
    ids: list[str], 
    descriptions: list[str],
    chroma_path: str,
    collection_name: str,
):

    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection(collection_name)
    logging.info("Initialized ChromaDB client")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logging.info("Initialized OpenAI client")

    full_descriptions = [
        md["title"] + "\n\n" + desc
        for md, desc in zip(metadatas, descriptions)
    ]

    logging.info("Embedding job descriptions...")
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=full_descriptions,
    )

    embeddings = [
        response.data[i].embedding for i in range(0, len(response.data))
    ]

    logging.info("Adding embeddings to collection...")
    collection.add(
        documents=full_descriptions,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    logging.info("Successfully vectorized jobs!")

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        type=str,
        help="Path to json file containing scraped jobs"
    )
    parser.add_argument(
        "--collection",
        type=str,
        help="Name of ChromaDB collection to store vector embeddings"
    )
    parser.add_argument(
        "--chromaPath",
        type=str,
        help="Path to save ChromaDB data"
    )
    args = parser.parse_args()

    init_logging()

    metadatas, ids, descriptions = parse_job_json(args.json)

    vectorize(metadatas, ids, descriptions, args.chromaPath, args.collection)
