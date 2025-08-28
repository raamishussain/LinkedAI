from linkedAI.agents.agent import Agent
from linkedAI.agents.data_models import (
    ResumeMatchArgs,
    ResumeMatchResult,
    ResumeTweakArgs,
    ResumeTweakResult,
)
from linkedAI.config import (
    DEFAULT_MODEL,
    RESUME_MATCH_PROMPT,
    RESUME_TWEAK_PROMPT,
)
from openai import OpenAI
from pathlib import Path
from pypdf import PdfReader
from typing import Any


class ResumeAgent(Agent):

    name = "ResumeAgent"

    def __init__(self, openai_client: OpenAI, resume_path: Path):

        super().__init__("ResumeAgent")

        self.client = openai_client
        self.model = DEFAULT_MODEL
        self.log("Initializing ResumeAgent...")

        try:
            self.resume = self._load_resume(resume_path)
            if not self.resume:
                self.log("Resume appears to be empty!", level="warning")
        except Exception as e:
            self.log(
                "Failed to initialize ResumeAgent",
                level="exception",
                resume_path=str(resume_path),
                exception=e,
            )
            raise

        self.log("Successfully initialized ResumeAgent")

    def _load_resume(self, resume_path: Path) -> str:

        if not resume_path.exists():
            self.log("Resume file not found", level="error")
            return ""

        try:
            if resume_path.suffix.lower() == ".pdf":
                reader = PdfReader(str(resume_path))
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                return "\n".join(text_parts)
            else:
                return resume_path.read_text(encoding="utf-8")
        except Exception as e:
            self.log(
                f"Failed to load resume from {resume_path}: {e}", level="error"
            )
            return ""

    @staticmethod
    def resume_match_tool_schema() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "match_job_to_resume",
                "description": "Compare the user's resume against a list of jobs and find the best match.",  # noqa: E501
                "parameters": ResumeMatchArgs.model_json_schema(),
            },
        }

    @staticmethod
    def resume_tweak_tool_schema() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "suggest_resume_tweaks",
                "description": "Suggest tweaks to the user's resume to better fit a given job.",  # noqa: E501
                "parameters": ResumeTweakArgs.model_json_schema(),
            },
        }

    def run_resume_match(
        self, resume_match_args: ResumeMatchArgs
    ) -> ResumeMatchResult:

        prompt = RESUME_MATCH_PROMPT.format(
            resume=self.resume,
            jobs=resume_match_args.jobs.model_dump_json(indent=2),
        )

        self.log("Sending resume match request to OpenAI...")
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=ResumeMatchResult,
            )
        except Exception as e:
            self.log(
                "Encountered exception during OpenAI API Call",
                level="error",
                exception=e,
            )

        result = response.choices[0].message.parsed

        if result is None:
            self.log(
                "OpenAI returned null result for ResumeMatchResult",
                level="warning",
            )
            result = ResumeMatchResult(best_match_id=0, reasoning="")

        self.log(
            "Resume matching completed successfully",
            best_match_id=result.best_match_id,
        )

        return result

    def run_resume_tweak(
        self, resume_tweak_args: ResumeTweakArgs
    ) -> ResumeTweakResult:

        prompt = RESUME_TWEAK_PROMPT.format(
            resume=self.resume,
            job_description=resume_tweak_args.job.description,
        )

        self.log("Sending resume tweak request to OpenAI...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            self.log(
                "Encountered exception during OpenAI API call",
                level="error",
                exception=e,
            )

        suggestions = response.choices[0].message.content or ""
        if suggestions:
            self.log("Resume tweak completed successfully")
        else:
            self.log("Resume tweak returned empty suggestion", level="warning")

        return ResumeTweakResult(suggestions=suggestions)
