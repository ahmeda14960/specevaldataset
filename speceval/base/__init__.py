"""Base abstract classes and fundamental types for the SpecEval framework."""
from .model import BaseModel, CandidateModel, EvaluatorModel, JudgeModel, RankingModel, BatchedModel
from .organization import Organization
from .parser import SpecificationParser, Specification
from .pipeline import Pipeline, TestCase, EvaluationResults
from .statement import Statement, StatementType, AuthorityLevel

__all__ = [
    "BaseModel",
    "CandidateModel",
    "EvaluatorModel",
    "JudgeModel",
    "RankingModel",
    "BatchedModel",
    "Organization",
    "SpecificationParser",
    "Specification",
    "Pipeline",
    "TestCase",
    "EvaluationResults",
    "Statement",
    "StatementType",
    "AuthorityLevel",
]
