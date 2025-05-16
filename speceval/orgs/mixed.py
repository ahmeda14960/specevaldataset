"""Mixed organization implementation for the SpecEval framework.

This implementation allows mixing models from different providers.
"""

from typing import Dict, Any, Optional

from ..base import Organization, CandidateModel, EvaluatorModel, JudgeModel
from ..models import (
    OpenAICandidateModel,
    OpenAIEvaluatorModel,
    OpenAIJudgeModel,
    AnthropicCandidateModel,
    AnthropicEvaluatorModel,
    AnthropicJudgeModel,
    TogetherCandidateModel,
)
from together import Together


class MixedOrganization(Organization):
    """
    Mixed organization implementation allowing any combination of models.

    This class allows creating an organization with any combination of models
    from different providers for candidate, evaluator, and judge roles.

    #TODO: Together Candidate Model shouldn't need the client to be passed in.
    """

    def __init__(
        self,
        candidate_provider: str = "openai",
        candidate_model_name: str = "gpt-4o-2024-08-06",
        evaluator_provider: str = "openai",
        evaluator_model_name: str = "gpt-4o-2024-08-06",
        judge_provider: str = "openai",
        judge_model_name: str = "gpt-4o-2024-08-06",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        together_api_key: Optional[str] = None,
        model_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize a mixed organization with models from different providers.

        Args:
            candidate_provider: Provider for candidate model ('openai', 'anthropic', or 'together').
            candidate_model_name: Name of the candidate model to use.
            evaluator_provider: Provider for evaluator model ('openai' or 'anthropic').
            evaluator_model_name: Name of the evaluator model to use.
            judge_provider: Provider for judge model ('openai' or 'anthropic').
            judge_model_name: Name of the judge model to use.
            openai_api_key: Optional API key for OpenAI models.
            anthropic_api_key: Optional API key for Anthropic models.
            together_api_key: Optional API key for Together models.
            model_kwargs: Optional dictionary with additional kwargs for each model type.
                          Keys should be 'candidate', 'evaluator', and 'judge'.
        """
        self.candidate_provider = candidate_provider.lower()
        self.candidate_model_name = candidate_model_name
        self.evaluator_provider = evaluator_provider.lower()
        self.evaluator_model_name = evaluator_model_name
        self.judge_provider = judge_provider.lower()
        self.judge_model_name = judge_model_name

        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.together_api_key = together_api_key

        self.model_kwargs = model_kwargs or {}

        # Initialize Together client if needed
        self.together_client = None
        if self.candidate_provider == "together":
            self.together_client = Together(api_key=self.together_api_key)

    def get_candidate_model(self) -> CandidateModel:
        """Get the candidate model based on the specified provider."""
        kwargs = self.model_kwargs.get("candidate", {})

        if self.candidate_provider == "anthropic":
            return AnthropicCandidateModel(
                model_name=self.candidate_model_name, api_key=self.anthropic_api_key, **kwargs
            )
        elif self.candidate_provider == "together":
            if not self.together_client:
                self.together_client = Together(api_key=self.together_api_key)
            return TogetherCandidateModel(
                model_name=self.candidate_model_name, client=self.together_client
            )
        else:  # openai is the default
            return OpenAICandidateModel(
                model_name=self.candidate_model_name, api_key=self.openai_api_key, **kwargs
            )

    def get_evaluator_model(self) -> EvaluatorModel:
        """Get the evaluator model based on the specified provider."""
        kwargs = self.model_kwargs.get("evaluator", {})

        if self.evaluator_provider == "anthropic":
            return AnthropicEvaluatorModel(
                model_name=self.evaluator_model_name, api_key=self.anthropic_api_key, **kwargs
            )
        else:  # openai is the default
            return OpenAIEvaluatorModel(
                model_name=self.evaluator_model_name, api_key=self.openai_api_key, **kwargs
            )

    def get_judge_model(self) -> JudgeModel:
        """Get the judge model based on the specified provider."""
        kwargs = self.model_kwargs.get("judge", {})

        if self.judge_provider == "anthropic":
            return AnthropicJudgeModel(
                model_name=self.judge_model_name, api_key=self.anthropic_api_key, **kwargs
            )
        else:  # openai is the default
            return OpenAIJudgeModel(
                model_name=self.judge_model_name, api_key=self.openai_api_key, **kwargs
            )

    def get_info(self) -> Dict[str, Any]:
        """Get metadata about the organization."""
        return {
            "name": "Mixed",
            "candidate_model": f"{self.candidate_provider}/{self.candidate_model_name}",
            "evaluator_model": f"{self.evaluator_provider}/{self.evaluator_model_name}",
            "judge_model": f"{self.judge_provider}/{self.judge_model_name}",
        }
