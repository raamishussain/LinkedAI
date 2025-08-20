# LinkedIn Job Scraper

This module contains code to scrape jobs from LinkedIn according to keywords and certain filters. Once jobs are scraped, the `vectorize.py` script can be used to create vector embeddings of the job descriptions which will be stored in a Chroma database.

NOTE: LinkedIn does not allow programmatic access to their website and repeated scraping of their site can lead
to your IP getting banned. Use at your own risk. To reduce your risk of ban, scrape slowly and do not scrape large amounts at a time.

## Scraping Jobs

To scrape jobs you must first set up your config in a JSON file. Your config will need to specify keywords to search
as well as some optional filters you can use you narrow your results. An example config can be found in `config_example.json` which looks like the following:

```json
{
  "keywords": "data scientist",
  "location": "United States",
  "time_since_post": 604800,
  "remote": true,
  "max_results": 10,
  "experience_levels": [
    "mid_senior"
  ],
  "salary": "180k"
}
```

### Required fields

Below are the required fields for your config and their descriptions:

* `keywords` (str): keywords to search for a specific job

### Optional Fields

Below are the optional fields for your config:

* `location` (`str`): Can be any location, but will default to `United States`
* `time_since_post` (int): Represents the time (in seconds) since the job was posted. Must be between 1 and 2592000 seconds (1 month)
* `remote` (`bool`): whether the job is remote or not (defaults to `False`)
* `max_results` (`int`): number of results to return from LinkedIn scrape
* `experience_levels` (`list[str]`): enum of experience levels to search for (see `data_models.py` for full list)
* `salary` (str): Enum of salary range (see `data_models.py` for full list)

Once your config is set, you can simply run the scrape script with the following command:

```bash
python -m scrape.py --configPath /path/to/config/file --output /path/to/output/json/file
```

This will result in a JSON file which will contain all the scraped jobs.

## Vectorize and Store Job Descriptions

Once your jobs are scraped, you can create vector embeddings of the job descriptions and store them in a ChromaDB vector store. To do this, you will need to have a valid OpenAI API key, since the vector embeddings are created using `text-embedding-3-small`.

To vectorize the data, run the following command:

```bash
python -m vectorize.py --json /path/to/job/json/file --chromaPath /path/to/store/chromadb --collection /path/to/chroma/collection
```

Note that if you provide an existing ChromaDB collection, the vector embeddings will simply be added to the collection. If the collection doesn't exist, it will be created and then the vector embeddings will be stored. 

NOTE: Currently there is no de-duping of the collection which means if you scrape jobs once and store the vector embeddings, and at a later date perform another scrape and store more vector embeddings, there very well could be duplicate jobs in the database. De-duping is on the to-do list.