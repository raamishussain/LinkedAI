import json

from bs4 import BeautifulSoup
from linkedAI.scraper.data_models import Config, JobCard
from playwright.sync_api import sync_playwright
from urllib.parse import urlencode, quote_plus


def load_config(file_path: str) -> Config:
    with open(file_path, "r") as file:
        config_data = json.load(file)
    return Config(**config_data)


def build_search_url(config: Config, start=0):
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


def scrape_job_links_from_search_page(page_html: str):
    soup = BeautifulSoup(page_html, "html.parser")
    job_links = []

    for a_tag in soup.select("a.base-card__full-link"):
        href = a_tag.get("href")
        if href and "/jobs/view/" in href:
            job_links.append(href.split("?")[0])

    return list(set(job_links))


def scrape_job_details(url: str):
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


def main():
    config = load_config("./linkedAI/scraper/config_example.json")

    jobs = scrape_linkedin_jobs(config)

    with open("scraped_jobs_10.json", "w", encoding="utf-8") as f:
        json.dump([job.model_dump() for job in jobs], f, indent=2)


if __name__ == "__main__":

    main()
