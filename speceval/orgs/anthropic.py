"""Anthropic organization implementation for the SpecEval framework."""

from typing import Dict, Any, Optional

from ..base import Organization, CandidateModel, EvaluatorModel, JudgeModel
from ..models import (
    AnthropicCandidateModel,
    AnthropicEvaluatorModel,
    AnthropicJudgeModel,
    OpenAIEvaluatorModel,
    OpenAIJudgeModel,
    GoogleJudgeModel,
)


class Anthropic(Organization):
    """
    Anthropic organization implementation.

    By default, all models (candidate, evaluator, and judge) use claude-3-7-sonnet-20250219.
    """

    def __init__(
        self,
        candidate_model_name: str = "claude-3-7-sonnet-20250219",
        evaluator_model_name: str = "claude-3-7-sonnet-20250219",
        judge_model_name: str = "claude-3-7-sonnet-20250219",
        evaluator_provider: str = "anthropic",
        judge_provider: str = "anthropic",
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the Anthropic organization.

        Args:
            candidate_model_name: Name of the candidate model to use.
            evaluator_model_name: Name of the evaluator model to use.
            judge_model_name: Name of the judge model to use.
            evaluator_provider: Provider for evaluator model ('anthropic' or 'openai').
            judge_provider: Provider for judge model ('anthropic' or 'openai').
            api_key: Optional API key to use for Anthropic models.
            openai_api_key: Optional API key to use for OpenAI models.
            model_kwargs: Optional dictionary with additional kwargs for each model type.
                          Keys should be 'candidate', 'evaluator', and 'judge'.
        """
        self.candidate_model_name = candidate_model_name
        self.evaluator_model_name = evaluator_model_name
        self.judge_model_name = judge_model_name
        self.evaluator_provider = evaluator_provider
        self.judge_provider = judge_provider
        self.api_key = api_key
        self.openai_api_key = openai_api_key
        self.model_kwargs = model_kwargs or {}

    def get_candidate_model(self) -> CandidateModel:
        """Get the candidate model."""
        kwargs = self.model_kwargs.get("candidate", {})
        return AnthropicCandidateModel(
            model_name=self.candidate_model_name, api_key=self.api_key, **kwargs
        )

    def get_evaluator_model(self) -> EvaluatorModel:
        """Get the evaluator model."""
        kwargs = self.model_kwargs.get("evaluator", {})
        if self.evaluator_provider.lower() == "openai":
            return OpenAIEvaluatorModel(
                model_name=self.evaluator_model_name, api_key=self.openai_api_key, **kwargs
            )
        return AnthropicEvaluatorModel(
            model_name=self.evaluator_model_name, api_key=self.api_key, **kwargs
        )

    def get_judge_model(self) -> JudgeModel:
        """Get the judge model."""
        kwargs = self.model_kwargs.get("judge", {})
        if self.judge_provider.lower() == "openai":
            return OpenAIJudgeModel(
                model_name=self.judge_model_name, api_key=self.openai_api_key, **kwargs
            )
        elif self.judge_provider.lower() == "google":
            return GoogleJudgeModel(model_name=self.judge_model_name, **kwargs)
        else:
            return AnthropicJudgeModel(
                model_name=self.judge_model_name, api_key=self.api_key, **kwargs
            )

    def get_info(self) -> Dict[str, Any]:
        """Get metadata about the organization."""
        return {
            "name": "Anthropic",
            "candidate_model": self.candidate_model_name,
            "evaluator_model": f"{self.evaluator_provider}/{self.evaluator_model_name}",
            "judge_model": f"{self.judge_provider}/{self.judge_model_name}",
        }
