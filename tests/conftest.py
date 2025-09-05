import pytest
import shutil
import tempfile
import json

from unittest.mock import Mock
from pathlib import Path
from linkedAI.scraper.data_models import JobCard
from linkedAI.agents.data_models import ResumeMatchResult, SearchResults


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    client = Mock()

    # Mock embeddings response
    embedding_response = Mock()
    embedding_response.data = [Mock()]
    embedding_response.data[0].embedding = [0.1, 0.2, 0.3] * 512  # 1536 dims
    client.embeddings.create.return_value = embedding_response

    # Mock chat completions response
    completion_response = Mock()
    choice = Mock()
    choice.message.content = "Test response"
    choice.message.tool_calls = None
    completion_response.choices = [choice]
    client.chat.completions.create.return_value = completion_response

    # Mock structured completion response for resume matching
    structured_response = Mock()
    structured_choice = Mock()
    structured_choice.message.parsed = ResumeMatchResult(
        best_match_id=0, reasoning="Test reasoning"
    )
    structured_response.choices = [structured_choice]
    client.beta.chat.completions.parse.return_value = structured_response

    return client


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection for testing"""
    collection = Mock()
    collection.query.return_value = {
        "documents": [["Test job description"]],
        "metadatas": [
            [
                {
                    "title": "Test Job",
                    "company": "Test Company",
                    "location": "Test Location",
                    "link": "http://test.com",
                }
            ]
        ],
    }
    return collection


@pytest.fixture
def sample_job_card():
    """Sample JobCard for testing"""
    return JobCard(
        title="Software Engineer",
        company="Tech Corp",
        location="San Francisco, CA",
        description="We are looking for a software engineer...",
        link="https://linkedin.com/jobs/view/123456",
    )


@pytest.fixture
def sample_job_cards():
    """List of sample JobCards for testing"""
    return [
        JobCard(
            title="Software Engineer",
            company="Tech Corp",
            location="San Francisco, CA",
            description="We are looking for a software engineer...",
            link="https://www.linkedin.com/jobs/view/cool-job-123456",
        ),
        JobCard(
            title="Data Scientist",
            company="Data Inc",
            location="New York, NY",
            description="Join our data science team...",
            link="https://www.linkedin.com/jobs/view/chill-job-789012",
        ),
    ]


@pytest.fixture
def sample_search_results(sample_job_cards):
    """Sample SearchResults for testing"""
    return SearchResults(jobs=sample_job_cards)


@pytest.fixture
def temp_json_file(sample_job_cards):
    """Temporary JSON file with job data for testing"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        job_data = [job.model_dump() for job in sample_job_cards]
        json.dump(job_data, f, indent=2)
        return f.name


@pytest.fixture
def mock_playwright():
    """Mock Playwright for web scraping tests"""
    playwright = Mock()
    browser = Mock()
    page = Mock()

    # Configure page mock
    page.locator.return_value.text_content.return_value = "Test Content"
    page.locator.return_value.nth.return_value.text_content.return_value = (
        "Test Content"
    )
    page.content.return_value = "<html><body>Test HTML</body></html>"

    browser.new_page.return_value = page
    playwright.chromium.launch.return_value = browser

    return playwright


@pytest.fixture
def sample_html_content():
    """Sample HTML content for scraping tests"""
    return """ # noqa: E501
    <html>
    <body>
        <ul class="jobs-search__results-list">
            <li>
                <a class="base-card__full-link" href="https://linkedin.com/jobs/view/123456?refId=test">Job 1</a>.
            </li>
            <li>
                <a class="base-card__full-link" href="https://linkedin.com/jobs/view/789012?refId=test">Job 2</a>
            </li>
        </ul>
    </body>
    </html>
    """


@pytest.fixture
def datadir(tmp_path, request):
    """Fixture responsible for searching a dolfer with the same name of test
    module and, if available, moving all contents to a temp directory so
    tests can use them freely.
    https://stackoverflow.com/a/29631801/720522
    """
    test_path = Path(request.module.__package__) / request.module.__file__
    test_artifact_dir = test_path.parent / Path(test_path.stem)

    if test_artifact_dir.exists() and test_artifact_dir.is_dir():
        shutil.copytree(test_artifact_dir, tmp_path, dirs_exist_ok=True)

    return tmp_path
