from enum import Enum
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from typing import Optional


class ExperienceLevel(str, Enum):
    ENTRY_LEVEL = "entry_level"
    ASSOCIATE = "associate"
    MID_SENIOR = "mid_senior"
    DIRECTOR = "director"
    EXECUTIVE = "executive"

class Salary(str, Enum):
    S100K = "100k"
    S120K = "120k"
    S140K = "140k"
    S160K = "160k"
    S180K = "180k"
    S200K = "200k"


class Config(BaseModel):
    keywords: str = Field(
        ..., 
        description="Keywords to search for jobs, just as you would type them in the LinkedIn search bar"
    )
    location: str = "United States"
    time_since_post: int =  Field(
            None, 
            gt=1, 
            le=2592000, 
            description="Time since post in seconds, must be between 1 and 2592000 (30 days)"
        )
    remote: bool = False
    max_results: int = 10
    experience_levels: Optional[list[ExperienceLevel]] = None
    salary: Optional[Salary] = None

    _f_E: Optional[str] = PrivateAttr(default=None)
    _f_TPR: Optional[str] = PrivateAttr(default=None)
    _f_WT: Optional[str] = PrivateAttr(default=1)

    def model_post_init(self, __context):
        level_map = {
            ExperienceLevel.ENTRY_LEVEL: "1",
            ExperienceLevel.ASSOCIATE: "2",
            ExperienceLevel.MID_SENIOR: "3",
            ExperienceLevel.DIRECTOR: "4",
            ExperienceLevel.EXECUTIVE: "5",
        }
        salary_map = {
            Salary.S100K: "4",
            Salary.S120K: "5",
            Salary.S140K: "6",
            Salary.S160K: "7",
            Salary.S180K: "8",
            Salary.S200K: "9",
        }

        if self.experience_levels:
            self._f_E = ",".join(level_map[l] for l in self.experience_levels)
        if self.time_since_post:
            self._f_TPR = f"r{self.time_since_post}"
        if self.remote:
            self._f_WT = "2"
        if self.salary:
            self._f_SB2 = salary_map.get(self.salary)

    @field_validator("keywords")
    def validate_keywords(cls, v):
        if not v:
            raise ValueError("Keywords cannot be empty")
        return v