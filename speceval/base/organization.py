"""Organization interface for the SpecEval framework."""
from abc import ABC, abstractmethod
from typing import Dict, Any

from .model import CandidateModel, EvaluatorModel, JudgeModel


class Organization(ABC):
    """
    Abstract base class representing an organization with AI models.

    Each organization provides its own candidate model, evaluator model,
    and judge model for evaluation.
    """

    @abstractmethod
    def get_candidate_model(self) -> CandidateModel:
        """Get the organization's candidate model."""
        pass

    @abstractmethod
    def get_evaluator_model(self) -> EvaluatorModel:
        """Get the organization's evaluator model."""
        pass

    @abstractmethod
    def get_judge_model(self) -> JudgeModel:
        """Get the organization's judge model."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get metadata about the organization."""
        pass
