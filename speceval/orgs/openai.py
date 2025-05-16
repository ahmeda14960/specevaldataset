"""OpenAI organization implementation for the SpecEval framework."""

from typing import Dict, Any, Optional

from ..base import Organization, CandidateModel, EvaluatorModel, JudgeModel
from ..models import (
    OpenAICandidateModel,
    OpenAIEvaluatorModel,
    OpenAIJudgeModel,
    AnthropicEvaluatorModel,
    AnthropicJudgeModel,
    GoogleJudgeModel,
)

from ..models.openai import GPT_4_1


class OpenAI(Organization):
    """
    OpenAI organization implementation.

    By default, all models (candidate, evaluator, and judge) use GPT-4o-2024-08-06.
    """

    def __init__(
        self,
        candidate_model_name: str = GPT_4_1,
        evaluator_model_name: str = GPT_4_1,
        judge_model_name: str = GPT_4_1,
        evaluator_provider: str = "openai",
        judge_provider: str = "openai",
        api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the OpenAI organization.

        Args:
            candidate_model_name: Name of the candidate model to use.
            evaluator_model_name: Name of the evaluator model to use.
            judge_model_name: Name of the judge model to use.
            evaluator_provider: Provider for evaluator model ('openai' or 'anthropic').
            judge_provider: Provider for judge model ('openai', 'anthropic', or 'google').
            api_key: Optional API key to use for OpenAI models.
            anthropic_api_key: Optional API key to use for Anthropic models.
            google_api_key: Optional API key to use for Google models.
            model_kwargs: Optional dictionary with additional kwargs for each model type.
                          Keys should be 'candidate', 'evaluator', and 'judge'.
        """
        self.candidate_model_name = candidate_model_name
        self.evaluator_model_name = evaluator_model_name
        self.judge_model_name = judge_model_name
        self.evaluator_provider = evaluator_provider
        self.judge_provider = judge_provider
        self.api_key = api_key
        self.anthropic_api_key = anthropic_api_key
        self.google_api_key = google_api_key
        self.model_kwargs = model_kwargs or {}

    def get_candidate_model(self) -> CandidateModel:
        """Get the candidate model."""
        kwargs = self.model_kwargs.get("candidate", {})
        return OpenAICandidateModel(
            model_name=self.candidate_model_name, api_key=self.api_key, **kwargs
        )

    def get_evaluator_model(self) -> EvaluatorModel:
        """Get the evaluator model."""
        kwargs = self.model_kwargs.get("evaluator", {})
        if self.evaluator_provider.lower() == "anthropic":
            return AnthropicEvaluatorModel(
                model_name=self.evaluator_model_name, api_key=self.anthropic_api_key, **kwargs
            )
        return OpenAIEvaluatorModel(
            model_name=self.evaluator_model_name, api_key=self.api_key, **kwargs
        )

    def get_judge_model(self) -> JudgeModel:
        """Get the judge model."""
        kwargs = self.model_kwargs.get("judge", {})
        if self.judge_provider.lower() == "anthropic":
            return AnthropicJudgeModel(
                model_name=self.judge_model_name, api_key=self.anthropic_api_key, **kwargs
            )
        elif self.judge_provider.lower() == "google":
            return GoogleJudgeModel(model_name=self.judge_model_name, **kwargs)
        return OpenAIJudgeModel(model_name=self.judge_model_name, api_key=self.api_key, **kwargs)

    def get_info(self) -> Dict[str, Any]:
        """Get metadata about the organization."""
        return {
            "name": "OpenAI",
            "candidate_model": self.candidate_model_name,
            "evaluator_model": f"{self.evaluator_provider}/{self.evaluator_model_name}",
            "judge_model": f"{self.judge_provider}/{self.judge_model_name}",
        }
