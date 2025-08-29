# LinkedAI

![logo](linkedAI_logo.png)

AI-powered job search assistant that combines LinkedIn job scraping with intelligent resume optimization and job matching using LLMs.

## Features

- **🔍 Intelligent Job Search**: Query a vector database of LinkedIn job postings using natural language
- **📊 Resume Matching**: AI-powered analysis to find jobs that best match your resume
- **✨ Resume Optimization**: Get specific suggestions to improve your resume for target roles
- **💬 Chat Interface**: Interactive Gradio web interface with streaming responses
- **🤖 Multi-Agent Architecture**: Specialized agents for querying, resume analysis, and chat coordination

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key
- Your resume in PDF format

### Installation

1. Clone the repository and make sure you are in the root directory

2. Install dependencies into a virtual environment using uv (recommended):

```bash
uv venv
source .venv/bin/activate
uv sync
```


3. Create a `.env` file in the root directory with the following contents:

```
OPENAI_API_KEY=<YOUR_KEY_HERE>
```

4. Place your resume PDF in `linkedAI/` directory and update the path in `linkedAI/config.py`

### Initial Setup

Before using the chat interface, you need to scrape and vectorize job data:

1. **Scrape LinkedIn Jobs** (see `linkedAI/scraper/README.md` for details):

```bash
cd linkedAI/scraper
python scrape.py --configPath config.json --output jobs.json
```

2. **Vectorize Job Data**:

```bash
python vectorize.py --json jobs.json --chromaPath ../../jobsDB --collection default
```

### Launch the Application

Run the Gradio web interface:
```bash
python -m linkedAI.gradio
```

The application will be available at `http://127.0.0.1:7860`

## Usage Examples

- **Find Jobs**: "Find data scientist jobs in San Francisco focusing on machine learning"
- **Resume Analysis**: "Which of these jobs best matches my resume?"
- **Resume Optimization**: "How can I improve my resume for this role?"
- **Follow-up Questions**: Ask detailed questions about specific jobs or requirements

## Project Structure

```
LinkedAI/
├── linkedAI/
│   ├── agents/           # Multi-agent system
│   │   ├── chat_agent.py    # Main chat coordinator
│   │   ├── query_agent.py   # Job search functionality
│   │   ├── resume_agent.py  # Resume analysis
│   │   └── data_models.py   # Pydantic models
│   ├── scraper/          # LinkedIn scraping tools
│   ├── config.py         # Configuration and prompts
│   └── gradio.py         # Web interface
├── jobsDB/              # ChromaDB vector store
└── pyproject.toml       # Project dependencies
```

## Configuration

Key configuration options in `linkedAI/config.py`:
- `OPENAI_API_KEY`: Your OpenAI API key (set in .env)
- `RESUME_PATH`: Path to your PDF resume
- `CHROMA_COLLECTION`: Path to ChromaDB storage
- `DEFAULT_MODEL`: OpenAI model to use (default: gpt-4o-mini)

## Dependencies

Core dependencies:
- `openai` - LLM API access
- `chromadb` - Vector database
- `gradio` - Web interface
- `pydantic` - data validation
- `pypdf` - Resume parsing
- `playwright` - Web scraping

## Important Notes

- **LinkedIn Scraping**: Use responsibly to avoid IP bans. Scrape slowly and in small batches.
- **Data Privacy**: Conversations are not stored permanently
- **API Usage**: Application uses OpenAI's API for processing
- **Resume Privacy**: Your resume is processed locally and not shared externally

## License

This project is for educational and personal use only. Beware of LinkedIn's Terms of Service when scraping.