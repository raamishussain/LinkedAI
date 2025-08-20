import argparse
import json

from bs4 import BeautifulSoup
from linkedAI.scraper.data_models import Config, JobCard
from playwright.sync_api import sync_playwright
from urllib.parse import urlencode, quote_plus


def load_config(file_path: str) -> Config:
    """Load JSON config file containing keywords and filters for job search

    Args:
        file_path: str
            Path to JSON config file

    Returns:
        config: Config
            Config object containing keywords and filters for job search
    """

    with open(file_path, "r") as file:
        config_data = json.load(file)
    return Config(**config_data)


def build_search_url(config: Config, start=0) -> str:
    """Contructs the LinkedIn search URL based on the provided config

    Args:
        config: Config
            config object containing keywords and filters for job search
        start: int = 0
            determines start page when searching LinkedIn for job results

    Returns:
        url: str
            full search url with all keywords and filters
    """

    base_url = "https://www.linkedin.com/jobs/search"
    query = {
        "keywords": config.keywords,
        "location": config.location,
        "start": start,
        "f_WT": config._f_WT,
    }

    if config._f_E:
        query["f_E"] = config._f_E
    if config._f_TPR:
        query["f_TPR"] = config._f_TPR
    if config._f_SB2:
        query["f_SB2"] = config._f_SB2

    return base_url + "?" + urlencode(query, quote_via=quote_plus)


def scrape_job_links_from_search_page(page_html: str) -> list[str]:
    """Extracts job link for each job

    Args:
        page_html: str
            Scraped HTML for a given job search

    Returns:
        job_links: list[str]
            Unique list of individual job links
    """

    soup = BeautifulSoup(page_html, "html.parser")
    job_links = []

    for a_tag in soup.select("a.base-card__full-link"):
        href = a_tag.get("href")
        if href and "/jobs/view/" in href:
            job_links.append(href.split("?")[0])

    return list(set(job_links))


def scrape_job_details(url: str) -> JobCard:
    """Scrape job details and return a JobCard with all relevant info

    Args:
        url: str
            URL for each individual job

    Returns:
        job_card: JobCard
            JobCard object containing job info
    """

    with sync_playwright() as sp:
        browser = sp.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_selector("h1", timeout=60000)

        try:
            title = page.locator("h1").text_content().strip()
            company = (
                page.locator("a.topcard__org-name-link").text_content().strip()
            )
        except Exception as e:
            print(e)
            company = (
                page.locator("span.topcard__flavor")
                .nth(0)
                .text_content()
                .strip()
            )

        try:
            location = (
                page.locator("span.topcard__flavor--bullet")
                .text_content()
                .strip()
            )
        except Exception:
            location = ""

        try:
            description = (
                page.locator("div.description__text").text_content().strip()
            )
        except Exception:
            description = ""

        browser.close()
        return JobCard(
            title=title,
            company=company,
            location=location,
            description=description,
            link=url,
        )


def scrape_linkedin_jobs(config: Config) -> list[JobCard]:
    """Function which builds the search URL, scrapes jobs and extracts info
    Args:
        config: Config
            Config object containing keywords and filters for job search

    Returns:
        results: list[JobCard]
            List of job cards with relevant info for each job
    """

    url = build_search_url(config)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_selector("ul.jobs-search__results-list", timeout=60000)
        html = page.content()
        browser.close()

    job_links = scrape_job_links_from_search_page(html)
    print(f"Found {len(job_links)} job links")

    results = []
    for link in job_links[: config.max_results]:
        try:
            job_data = scrape_job_details(link)
            results.append(job_data)
        except Exception as e:
            print(f"failed to scrape {link}: {e}")

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--configPath",
        type=str,
        help="Path to JSON config file containing keywords and filters",
    )
    parser.add_argument(
        "--output", type=str, help="Path to JSON output file for scraped jobs"
    )
    args = parser.parse_args()

    config = load_config(args.configPath)

    jobs = scrape_linkedin_jobs(config)

    # dump results into a json file for now
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([job.model_dump() for job in jobs], f, indent=2)
