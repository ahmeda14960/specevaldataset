"""SpecEval: A framework for testing AI model compliance with policy specifications."""
from .base import (
    BaseModel,
    CandidateModel,
    EvaluatorModel,
    JudgeModel,
    Organization,
    SpecificationParser,
    Specification,
    Pipeline,
    TestCase,
    EvaluationResults,
    Statement,
    StatementType,
    AuthorityLevel,
)

from .parsers import MarkdownParser, JsonlParser
from .models import OpenAICandidateModel, OpenAIEvaluatorModel, OpenAIJudgeModel
from .orgs import OpenAI
from .autobencher import AutoBencher
from .pipelines import StandardPipeline

__all__ = [
    "BaseModel",
    "CandidateModel",
    "EvaluatorModel",
    "JudgeModel",
    "Organization",
    "SpecificationParser",
    "Specification",
    "Pipeline",
    "TestCase",
    "EvaluationResults",
    "Statement",
    "StatementType",
    "AuthorityLevel",
    "MarkdownParser",
    "OpenAICandidateModel",
    "OpenAIEvaluatorModel",
    "OpenAIJudgeModel",
    "OpenAI",
    "AutoBencher",
    "StandardPipeline",
    "JsonlParser",
]

__version__ = "0.1.0"
