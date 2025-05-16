"""Google organization implementation for the SpecEval framework."""

from typing import Dict, Any, Optional

from ..base import Organization, CandidateModel, EvaluatorModel, JudgeModel
from ..models import (
    GoogleCandidateModel,
    GoogleEvaluatorModel,
    GoogleJudgeModel,
    OpenAIEvaluatorModel,
    OpenAIJudgeModel,
    AnthropicEvaluatorModel,
    AnthropicJudgeModel,
)


class Google(Organization):
    """
    Google organization implementation.

    Allows using Google Gemini models as candidate, evaluator, or judge.
    Also supports mixing with OpenAI or Anthropic models for evaluator/judge roles.
    """

    def __init__(
        self,
        candidate_model_name: str = "gemini-2.0-flash-001",
        evaluator_model_name: str = "gemini-2.0-flash-001",
        judge_model_name: str = "gemini-2.0-flash-001",
        evaluator_provider: str = "google",
        judge_provider: str = "google",
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the Google organization.

        Args:
            candidate_model_name: Name of the candidate model (must be a Google model).
            evaluator_model_name: Name of the evaluator model to use.
            judge_model_name: Name of the judge model to use.
            evaluator_provider: Provider for evaluator model ('google', 'openai', or 'anthropic').
            judge_provider: Provider for judge model ('google', 'openai', or 'anthropic').
            api_key: Optional API key for Google models.
            openai_api_key: Optional API key for OpenAI models (if used).
            anthropic_api_key: Optional API key for Anthropic models (if used).
            model_kwargs: Optional dictionary with additional kwargs for each model type.
                          Keys should be 'candidate', 'evaluator', and 'judge'.
        """
        # Candidate must currently be Google
        self.candidate_model_name = candidate_model_name
        self.evaluator_model_name = evaluator_model_name
        self.judge_model_name = judge_model_name
        self.evaluator_provider = evaluator_provider.lower()
        self.judge_provider = judge_provider.lower()
        self.api_key = api_key  # Google API Key
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.model_kwargs = model_kwargs or {}

    def get_candidate_model(self) -> CandidateModel:
        """Get the candidate model (must be a Google model)."""
        kwargs = self.model_kwargs.get("candidate", {})
        # Currently, the candidate for the Google org must be a Google model
        return GoogleCandidateModel(
            model_name=self.candidate_model_name, api_key=self.api_key, **kwargs
        )

    def get_evaluator_model(self) -> EvaluatorModel:
        """Get the evaluator model based on the specified provider."""
        kwargs = self.model_kwargs.get("evaluator", {})
        if self.evaluator_provider == "openai":
            return OpenAIEvaluatorModel(
                model_name=self.evaluator_model_name, api_key=self.openai_api_key, **kwargs
            )
        elif self.evaluator_provider == "anthropic":
            return AnthropicEvaluatorModel(
                model_name=self.evaluator_model_name, api_key=self.anthropic_api_key, **kwargs
            )
        elif self.evaluator_provider == "google":
            return GoogleEvaluatorModel(
                model_name=self.evaluator_model_name, api_key=self.api_key, **kwargs
            )
        else:
            raise ValueError(f"Unsupported evaluator provider: {self.evaluator_provider}")

    def get_judge_model(self) -> JudgeModel:
        """Get the judge model based on the specified provider."""
        kwargs = self.model_kwargs.get("judge", {})
        if self.judge_provider == "openai":
            return OpenAIJudgeModel(
                model_name=self.judge_model_name, api_key=self.openai_api_key, **kwargs
            )
        elif self.judge_provider == "anthropic":
            return AnthropicJudgeModel(
                model_name=self.judge_model_name, api_key=self.anthropic_api_key, **kwargs
            )
        elif self.judge_provider == "google":
            return GoogleJudgeModel(
                model_name=self.judge_model_name, api_key=self.api_key, **kwargs
            )
        else:
            raise ValueError(f"Unsupported judge provider: {self.judge_provider}")

    def get_info(self) -> Dict[str, Any]:
        """Get metadata about the organization configuration."""
        return {
            "name": "Google",
            "candidate_model": self.candidate_model_name,  # Assuming candidate is always Google
            "evaluator_model": f"{self.evaluator_provider}/{self.evaluator_model_name}",
            "judge_model": f"{self.judge_provider}/{self.judge_model_name}",
        }
