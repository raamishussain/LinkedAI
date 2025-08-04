import chromadb
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def parse_job_json(json_path: str):

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

        ids.append(
            job["link"].split("-")[-1]
        )  # get job id from end of job link\

        descriptions.append(job["description"])

    return metadatas, ids, descriptions


def vectorize(metadatas, ids, descriptions):

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection("linkedIn_jobs_20250803")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    full_descriptions = [
        md["title"] + "\n\n" + desc
        for md, desc in zip(metadatas, descriptions)
    ]

    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=full_descriptions,
    )

    embeddings = [
        response.data[i].embedding for i in range(0, len(response.data))
    ]

    collection.add(
        documents=full_descriptions,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
