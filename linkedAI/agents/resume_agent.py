import logging

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
from pydantic import ValidationError
from typing import Any


class ResumeAgent(Agent):

    name = "ResumeAgent"

    def __init__(self, openai_client: OpenAI, resume_path: Path):

        self.client = openai_client
        self.model = DEFAULT_MODEL
        self.resume = self._load_resume(resume_path)
        self.log("Successfully initialized ResumeAgent")

    @staticmethod
    def _load_resume(resume_path: str) -> str:
        try:
            text = Path(resume_path).read_text(encoding="utf-8")
            return text
        except Exception as e:
            logging.error(f"Failed to load resume: {e}")
            return ""

    def resume_match_tool_schema() -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "match_job_to_resume",
                "description": "Compare the user's resume against a list of jobs and find the best match.",  # noqa: E501
                "parameters": ResumeMatchArgs.model_json_schema(),
            },
        }

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        try:
            data = ResumeMatchResult.model_validate_json(content)
        except ValidationError:
            self.log("Resume matching response failed validation")
            data = ResumeMatchResult(best_match_id=0, reasoning="")

        return data

    def run_resume_tweak(
        self, resume_tweak_args: ResumeTweakArgs
    ) -> ResumeTweakResult:

        prompt = RESUME_TWEAK_PROMPT.format(
            resume=self.resume,
            job_description=resume_tweak_args.job.description,
        )

        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )

        suggestions = response.choices[0].message.content or ""
        return ResumeTweakResult(suggestions=suggestions)
