import pytest
from unittest.mock import Mock, patch

from linkedAI.scraper.scrape import (
    load_config,
    build_search_url,
    scrape_job_links_from_search_page,
    scrape_job_details,
    scrape_linkedin_jobs,
)
from linkedAI.scraper.vectorize import (
    init_logging,
    parse_job_json,
    vectorize,
)
from linkedAI.scraper.data_models import (
    Config,
    JobCard,
    ExperienceLevel,
    Salary,
)


class TestScraperDataModels:
    """Test scraper data models"""

    def test_job_card_creation(self):
        job = JobCard(
            title="Software Engineer",
            company="Tech Corp",
            location="San Francisco, CA",
            description="We are looking for a software engineer...",
            link="https://linkedin.com/jobs/view/123456",
        )

        assert job.title == "Software Engineer"
        assert job.company == "Tech Corp"
        assert job.location == "San Francisco, CA"
        assert job.description == "We are looking for a software engineer..."
        assert job.link == "https://linkedin.com/jobs/view/123456"

    def test_config_basic_creation(self):
        config = Config(keywords="python developer")
        assert config.keywords == "python developer"
        assert config.location == "United States"
        assert config.max_results == 10
        assert config._f_WT == "1"

    def test_config_with_experience_levels(self):
        config = Config(
            keywords="senior engineer",
            experience_levels=[
                ExperienceLevel.MID_SENIOR,
                ExperienceLevel.DIRECTOR,
            ],
        )
        assert config._f_E == "4,5"

    def test_config_with_remote(self):
        config = Config(keywords="remote developer", remote=True)
        assert config._f_WT == "2"  # Remote work type

    def test_config_with_salary(self):
        config = Config(keywords="high paying job", salary=Salary.S160K)
        assert config._f_SB2 == "7"  # $160k maps to "7"

    def test_config_with_time_since_post(self):
        config = Config(keywords="recent jobs", time_since_post=86400)
        assert config._f_TPR == "r86400"

    def test_config_empty_keywords_validation(self):
        with pytest.raises(ValueError, match="Keywords cannot be empty"):
            Config(keywords="")

    def test_config_time_since_post_validation(self):
        # Test minimum boundary
        with pytest.raises(ValueError):
            Config(keywords="test", time_since_post=0)

        # Test maximum boundary
        with pytest.raises(ValueError):
            Config(keywords="test", time_since_post=2592001)

    def test_experience_level_enum(self):
        assert ExperienceLevel.INTERNSHIP == "internship"
        assert ExperienceLevel.ENTRY_LEVEL == "entry_level"
        assert ExperienceLevel.MID_SENIOR == "mid_senior"

    def test_salary_enum(self):
        assert Salary.S100K == "100k"
        assert Salary.S200K == "200k"


class TestScrapeModule:
    """Test scraping functions"""

    def test_load_config(self, datadir):
        config = load_config(datadir / "test_scrape_config.json")

        assert config.keywords == "data scientist"
        assert config.location == "United States"
        assert config.max_results == 10
        assert config.remote
        assert config.experience_levels == ["mid_senior"]
        assert config.salary == "180k"

    def test_build_search_url_basic(self, datadir):

        config = load_config(datadir / "test_scrape_config.json")
        url = build_search_url(config)

        assert "https://www.linkedin.com/jobs/search" in url
        assert "keywords=data+scientist" in url
        assert "location=United+States" in url
        assert "start=0" in url
        assert "f_WT=2" in url
        assert "f_E=4" in url
        assert "f_TPR=r604800" in url
        assert "f_SB2=8" in url

    def test_scrape_job_links_from_search_page(self, sample_html_content):
        job_links = scrape_job_links_from_search_page(sample_html_content)

        assert len(job_links) == 2
        assert "https://linkedin.com/jobs/view/123456" in job_links
        assert "https://linkedin.com/jobs/view/789012" in job_links

        # Check that query parameters are stripped
        for link in job_links:
            assert "?refId" not in link

    def test_scrape_job_links_no_jobs(self):
        html_content = "<html><body><p>No jobs found</p></body></html>"
        job_links = scrape_job_links_from_search_page(html_content)
        assert job_links == []

    def test_scrape_job_links_duplicates_removed(self):
        html_content = """ # noqa: E501
        <html><body>
            <a class="base-card__full-link" href="https://linkedin.com/jobs/view/123456?ref1">Job 1</a>
            <a class="base-card__full-link" href="https://linkedin.com/jobs/view/123456?ref2">Job 1 Again</a>
        </body></html>
        """
        job_links = scrape_job_links_from_search_page(html_content)
        assert len(job_links) == 1
        assert job_links[0] == "https://linkedin.com/jobs/view/123456"

    @patch("linkedAI.scraper.scrape.sync_playwright")
    def test_scrape_job_details_success(self, mock_playwright):
        # Setup mock playwright
        mock_context = mock_playwright.return_value.__enter__.return_value
        mock_browser = Mock()
        mock_page = Mock()
        mock_browser.new_page.return_value = mock_page
        mock_context.chromium.launch.return_value = mock_browser

        # Configure page locators
        mock_page.locator.return_value.text_content.return_value.strip.return_value = (  # noqa: E501
            "Mock Content"
        )

        # Configure specific locators
        def locator_side_effect(selector):
            mock_locator = Mock()
            if "h1" in selector:
                mock_locator.text_content.return_value.strip.return_value = (
                    "Software Engineer"
                )
            elif "topcard__org-name-link" in selector:
                mock_locator.text_content.return_value.strip.return_value = (
                    "Tech Corp"
                )
            elif "topcard__flavor--bullet" in selector:
                mock_locator.text_content.return_value.strip.return_value = (
                    "San Francisco, CA"
                )
            elif "description__text" in selector:
                mock_locator.text_content.return_value.strip.return_value = (
                    "Job description here"
                )
            else:
                mock_locator.text_content.return_value.strip.return_value = (
                    "Default Content"
                )
            return mock_locator

        mock_page.locator.side_effect = locator_side_effect

        url = "https://linkedin.com/jobs/view/123456"
        job_card = scrape_job_details(url)

        assert isinstance(job_card, JobCard)
        assert job_card.title == "Software Engineer"
        assert job_card.company == "Tech Corp"
        assert job_card.location == "San Francisco, CA"
        assert job_card.description == "Job description here"
        assert job_card.link == url

        mock_page.goto.assert_called_once_with(url, timeout=60000)
        mock_page.wait_for_selector.assert_called_once_with(
            "h1", timeout=60000
        )
        mock_browser.close.assert_called_once()

    @patch("linkedAI.scraper.scrape.sync_playwright")
    def test_scrape_job_details_missing_fields(self, mock_playwright):
        # Setup mock playwright
        mock_context = mock_playwright.return_value.__enter__.return_value
        mock_browser = Mock()
        mock_page = Mock()
        mock_browser.new_page.return_value = mock_page
        mock_context.chromium.launch.return_value = mock_browser

        # Configure locators to raise exceptions for optional fields
        def locator_side_effect(selector):
            mock_locator = Mock()
            if "h1" in selector:
                mock_locator.text_content.return_value.strip.return_value = (
                    "Software Engineer"
                )
            elif "topcard__org-name-link" in selector:
                mock_locator.text_content.return_value.strip.return_value = (
                    "Tech Corp"
                )
            elif "topcard__flavor--bullet" in selector:
                mock_locator.text_content.side_effect = Exception("Not found")
            elif "description__text" in selector:
                mock_locator.text_content.side_effect = Exception("Not found")
            else:
                mock_locator.text_content.return_value.strip.return_value = (
                    "Default"
                )
            return mock_locator

        mock_page.locator.side_effect = locator_side_effect

        url = "https://linkedin.com/jobs/view/123456"
        job_card = scrape_job_details(url)

        assert job_card.title == "Software Engineer"
        assert job_card.company == "Tech Corp"
        assert job_card.location == ""  # Should be empty when not found
        assert job_card.description == ""  # Should be empty when not found

    @patch("linkedAI.scraper.scrape.scrape_job_links_from_search_page")
    @patch("linkedAI.scraper.scrape.scrape_job_details")
    @patch("linkedAI.scraper.scrape.sync_playwright")
    def test_scrape_linkedin_jobs(
        self,
        mock_playwright,
        mock_scrape_details,
        mock_scrape_links,
        datadir,
    ):
        # Setup playwright mock
        mock_context = mock_playwright.return_value.__enter__.return_value
        mock_browser = Mock()
        mock_page = Mock()
        mock_browser.new_page.return_value = mock_page
        mock_context.chromium.launch.return_value = mock_browser
        mock_page.content.return_value = "<html>Mock HTML</html>"

        # Setup scraping mocks
        mock_scrape_links.return_value = [
            "https://linkedin.com/jobs/view/123456",
            "https://linkedin.com/jobs/view/789012",
        ]

        mock_job_cards = [
            JobCard(
                title="Engineer 1",
                company="Company 1",
                location="Location 1",
                description="Description 1",
                link="https://linkedin.com/jobs/view/123456",
            ),
            JobCard(
                title="Engineer 2",
                company="Company 2",
                location="Location 2",
                description="Description 2",
                link="https://linkedin.com/jobs/view/789012",
            ),
        ]
        mock_scrape_details.side_effect = mock_job_cards

        config = load_config(datadir / "test_scrape_config.json")

        results = scrape_linkedin_jobs(config)

        assert len(results) == 2
        assert all(isinstance(job, JobCard) for job in results)
        assert results[0].title == "Engineer 1"
        assert results[1].title == "Engineer 2"

        mock_page.goto.assert_called_once()
        mock_page.wait_for_selector.assert_called_once_with(
            "ul.jobs-search__results-list", timeout=60000
        )
        mock_browser.close.assert_called_once()

    @patch("linkedAI.scraper.scrape.scrape_job_links_from_search_page")
    @patch("linkedAI.scraper.scrape.scrape_job_details")
    @patch("linkedAI.scraper.scrape.sync_playwright")
    def test_scrape_linkedin_jobs_with_max_results(
        self, mock_playwright, mock_scrape_details, mock_scrape_links
    ):
        # Setup mocks
        mock_context = mock_playwright.return_value.__enter__.return_value
        mock_browser = Mock()
        mock_page = Mock()
        mock_browser.new_page.return_value = mock_page
        mock_context.chromium.launch.return_value = mock_browser
        mock_page.content.return_value = "<html>Mock HTML</html>"

        # Return more links than max_results
        mock_scrape_links.return_value = [
            "https://linkedin.com/jobs/view/1",
            "https://linkedin.com/jobs/view/2",
            "https://linkedin.com/jobs/view/3",
            "https://linkedin.com/jobs/view/4",
            "https://linkedin.com/jobs/view/5",
        ]

        mock_job_card = JobCard(
            title="Engineer",
            company="Company",
            location="Location",
            description="Description",
            link="test-link",
        )
        mock_scrape_details.return_value = mock_job_card

        config = Config(keywords="test", max_results=3)
        results = scrape_linkedin_jobs(config)

        assert len(results) == 3  # Should be limited by max_results
        assert mock_scrape_details.call_count == 3

    @patch("linkedAI.scraper.scrape.scrape_job_links_from_search_page")
    @patch("linkedAI.scraper.scrape.scrape_job_details")
    @patch("linkedAI.scraper.scrape.sync_playwright")
    def test_scrape_linkedin_jobs_with_errors(
        self, mock_playwright, mock_scrape_details, mock_scrape_links, capfd
    ):
        # Setup mocks
        mock_context = mock_playwright.return_value.__enter__.return_value
        mock_browser = Mock()
        mock_page = Mock()
        mock_browser.new_page.return_value = mock_page
        mock_context.chromium.launch.return_value = mock_browser
        mock_page.content.return_value = "<html>Mock HTML</html>"

        mock_scrape_links.return_value = [
            "https://linkedin.com/jobs/view/123456",
            "https://linkedin.com/jobs/view/789012",
        ]

        # First call succeeds, second fails
        mock_job_card = JobCard(
            title="Engineer",
            company="Company",
            location="Location",
            description="Description",
            link="https://linkedin.com/jobs/view/123456",
        )
        mock_scrape_details.side_effect = [
            mock_job_card,
            Exception("Scraping failed"),
        ]

        config = Config(keywords="test", max_results=5)
        results = scrape_linkedin_jobs(config)

        assert len(results) == 1  # Only successful scraping
        captured = capfd.readouterr()
        assert "failed to scrape" in captured.out


class TestVectorizeModule:
    """Test vectorization functions"""

    def test_init_logging(self, caplog):
        init_logging()

        import logging

        logger = logging.getLogger()
        logger.info("Test log message")

        assert "Test log message" in caplog.text

    def test_parse_job_json(self, temp_json_file):
        metadatas, ids, descriptions = parse_job_json(temp_json_file)

        assert len(metadatas) == 2
        assert len(ids) == 2
        assert len(descriptions) == 2

        assert metadatas[0]["title"] == "Software Engineer"
        assert metadatas[0]["company"] == "Tech Corp"
        assert metadatas[0]["location"] == "San Francisco, CA"
        assert (
            metadatas[0]["link"]
            == "https://www.linkedin.com/jobs/view/cool-job-123456"
        )
        assert ids[0] == "123456"  # Extracted from end of link
        assert "software engineer" in descriptions[0].lower()

    @patch("chromadb.PersistentClient")
    @patch("linkedAI.scraper.vectorize.OpenAI")
    def test_vectorize_success(
        self, mock_openai_class, mock_chroma_client, mock_openai_client, caplog
    ):
        # Setup mocks
        mock_openai_class.return_value = mock_openai_client
        mock_collection = Mock()
        mock_chroma_instance = Mock()
        mock_chroma_instance.get_or_create_collection.return_value = (
            mock_collection
        )
        mock_chroma_client.return_value = mock_chroma_instance

        # Setup sample data
        metadatas = [
            {
                "title": "Engineer",
                "company": "Corp",
                "location": "SF",
                "link": "http://test.com",
            }
        ]
        ids = ["123456"]
        descriptions = ["Job description here"]

        vectorize(
            metadatas, ids, descriptions, "/test/path", "test_collection"
        )

        # Verify ChromaDB setup
        mock_chroma_client.assert_called_once_with(path="/test/path")
        mock_chroma_instance.get_or_create_collection.assert_called_once_with(
            "test_collection"
        )

        # Verify OpenAI embedding call
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["Engineer\n\nJob description here"],  # Title + description
        )

        # Verify data added to collection
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args
        assert call_args[1]["documents"] == [
            "Engineer\n\nJob description here"
        ]
        assert call_args[1]["metadatas"] == metadatas
        assert call_args[1]["ids"] == ids
        assert "embeddings" in call_args[1]

        # Check logging
        assert "Initialized ChromaDB client" in caplog.text
        assert "Initialized OpenAI client" in caplog.text
        assert "Embedding job descriptions..." in caplog.text
        assert "Successfully vectorized jobs!" in caplog.text

    @patch("chromadb.PersistentClient")
    @patch("linkedAI.scraper.vectorize.OpenAI")
    def test_vectorize_multiple_jobs(
        self, mock_openai_class, mock_chroma_client, mock_openai_client
    ):
        # Setup mocks
        mock_openai_class.return_value = mock_openai_client
        mock_collection = Mock()
        mock_chroma_instance = Mock()
        mock_chroma_instance.get_or_create_collection.return_value = (
            mock_collection
        )
        mock_chroma_client.return_value = mock_chroma_instance

        # Mock embedding response with multiple embeddings
        embedding_data = [Mock(), Mock()]
        embedding_data[0].embedding = [0.1] * 1536
        embedding_data[1].embedding = [0.2] * 1536
        mock_openai_client.embeddings.create.return_value.data = embedding_data

        # Setup sample data
        metadatas = [
            {
                "title": "Engineer 1",
                "company": "Corp 1",
                "location": "SF",
                "link": "http://test1.com",
            },
            {
                "title": "Engineer 2",
                "company": "Corp 2",
                "location": "NY",
                "link": "http://test2.com",
            },
        ]
        ids = ["123456", "789012"]
        descriptions = ["Description 1", "Description 2"]

        vectorize(
            metadatas, ids, descriptions, "/test/path", "test_collection"
        )

        # Verify embedding call with combined titles and descriptions
        expected_input = [
            "Engineer 1\n\nDescription 1",
            "Engineer 2\n\nDescription 2",
        ]
        mock_openai_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small", input=expected_input
        )

        # Verify collection.add called with correct embeddings
        call_args = mock_collection.add.call_args
        embeddings = call_args[1]["embeddings"]
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1] * 1536
        assert embeddings[1] == [0.2] * 1536
