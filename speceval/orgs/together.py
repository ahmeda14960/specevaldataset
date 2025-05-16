"""Together AI organization implementations for the SpecEval framework."""
from typing import Dict, Any, Optional
from together import Together

from ..base import Organization, CandidateModel, EvaluatorModel, JudgeModel
from ..models.openai import OpenAIEvaluatorModel, OpenAIJudgeModel
from ..models.together import TogetherCandidateModel
from ..models import AnthropicJudgeModel, GoogleJudgeModel


class MetaOrganization(Organization):
    """Meta organization using Llama 3."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        judge_provider: str = "openai",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        judge_model_name: str = "gpt-4.1-2025-04-14",
    ):
        """
        Initialize the Meta organization with Llama 3 model.

        Args:
            api_key: Optional API key for Together AI.
            judge_provider: Provider for judge model ('openai', 'anthropic', or 'google').
            openai_api_key: Optional API key for OpenAI models.
            anthropic_api_key: Optional API key for Anthropic models.
            judge_model_name: Name of the judge model to use.
        """
        self.client = Together(api_key=api_key)
        self.candidate_model = TogetherCandidateModel(
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Lite-Pro", self.client
        )
        self.evaluator_model = OpenAIEvaluatorModel(api_key=openai_api_key)
        print(f"evaluator_model is : {self.evaluator_model.model_name}")

        self.judge_provider = judge_provider
        self.judge_model_name = judge_model_name
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key

        if judge_provider.lower() == "openai":
            self.judge_model = OpenAIJudgeModel(model_name=judge_model_name, api_key=openai_api_key)
        elif judge_provider.lower() == "anthropic":
            self.judge_model = AnthropicJudgeModel(
                model_name=judge_model_name, api_key=anthropic_api_key
            )
        elif judge_provider.lower() == "google":
            self.judge_model = GoogleJudgeModel(model_name=judge_model_name)
        else:
            self.judge_model = OpenAIJudgeModel(api_key=openai_api_key)

        print(f"judge_model is : {self.judge_model.model_name}")

    def get_candidate_model(self) -> CandidateModel:
        """Get the candidate model."""
        return self.candidate_model

    def get_evaluator_model(self) -> EvaluatorModel:
        """Get the evaluator model."""
        return self.evaluator_model

    def get_judge_model(self) -> JudgeModel:
        """Get the judge model."""
        return self.judge_model

    def get_info(self) -> Dict[str, Any]:
        """Get metadata about the organization."""
        return {
            "name": "Meta",
            "candidate_model": self.candidate_model.get_info(),
            "evaluator_model": self.evaluator_model.get_info(),
            "judge_model": f"{self.judge_provider}/{self.judge_model.model_name}",
        }


class QwenOrganization(Organization):
    """Qwen organization using Qwen 72B."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        judge_provider: str = "openai",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        judge_model_name: str = "gpt-4.1-2025-04-14",
    ):
        """
        Initialize the Qwen organization with Qwen 72B model.

        Args:
            api_key: Optional API key for Together AI.
            judge_provider: Provider for judge model ('openai', 'anthropic', or 'google').
            openai_api_key: Optional API key for OpenAI models.
            anthropic_api_key: Optional API key for Anthropic models.
            judge_model_name: Name of the judge model to use.
        """
        self.client = Together(api_key=api_key)
        self.candidate_model = TogetherCandidateModel(
            "Qwen/Qwen2.5-72B-Instruct-Turbo", self.client
        )

        self.evaluator_model = OpenAIEvaluatorModel(api_key=openai_api_key)
        print(f"evaluator_model is : {self.evaluator_model.model_name}")

        self.judge_provider = judge_provider
        self.judge_model_name = judge_model_name
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key

        if judge_provider.lower() == "openai":
            self.judge_model = OpenAIJudgeModel(model_name=judge_model_name, api_key=openai_api_key)
        elif judge_provider.lower() == "anthropic":
            self.judge_model = AnthropicJudgeModel(
                model_name=judge_model_name, api_key=anthropic_api_key
            )
        elif judge_provider.lower() == "google":
            self.judge_model = GoogleJudgeModel(model_name=judge_model_name)
        else:
            self.judge_model = OpenAIJudgeModel(api_key=openai_api_key)

        print(f"judge_model is : {self.judge_model.model_name}")

    def get_candidate_model(self) -> CandidateModel:
        """Get the candidate model."""
        return self.candidate_model

    def get_evaluator_model(self) -> EvaluatorModel:
        """Get the evaluator model."""
        return self.evaluator_model

    def get_judge_model(self) -> JudgeModel:
        """Get the judge model."""
        return self.judge_model

    def get_info(self) -> Dict[str, Any]:
        """Get metadata about the organization."""
        return {
            "name": "Qwen",
            "candidate_model": self.candidate_model.get_info(),
            "evaluator_model": self.evaluator_model.get_info(),
            "judge_model": f"{self.judge_provider}/{self.judge_model.model_name}",
        }


class DeepSeekOrganization(Organization):
    """DeepSeek organization using DeepSeek models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        judge_provider: str = "openai",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        judge_model_name: str = "gpt-4.1-2025-04-14",
    ):
        """
        Initialize the DeepSeek organization with DeepSeek model.

        Args:
            api_key: Optional API key for Together AI.
            judge_provider: Provider for judge model ('openai', 'anthropic', or 'google').
            openai_api_key: Optional API key for OpenAI models.
            anthropic_api_key: Optional API key for Anthropic models.
            judge_model_name: Name of the judge model to use.
        """
        self.client = Together(api_key=api_key)
        self.candidate_model = TogetherCandidateModel("deepseek-ai/DeepSeek-R1", self.client)
        self.evaluator_model = OpenAIEvaluatorModel(api_key=openai_api_key)
        print(f"evaluator_model is : {self.evaluator_model.model_name}")

        self.judge_provider = judge_provider
        self.judge_model_name = judge_model_name
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key

        if judge_provider.lower() == "openai":
            self.judge_model = OpenAIJudgeModel(model_name=judge_model_name, api_key=openai_api_key)
        elif judge_provider.lower() == "anthropic":
            self.judge_model = AnthropicJudgeModel(
                model_name=judge_model_name, api_key=anthropic_api_key
            )
        elif judge_provider.lower() == "google":
            self.judge_model = GoogleJudgeModel(model_name=judge_model_name)
        else:
            self.judge_model = OpenAIJudgeModel(api_key=openai_api_key)

        print(f"judge_model is : {self.judge_model.model_name}")

    def get_candidate_model(self) -> CandidateModel:
        """Get the candidate model."""
        return self.candidate_model

    def get_evaluator_model(self) -> EvaluatorModel:
        """Get the evaluator model."""
        return self.evaluator_model

    def get_judge_model(self) -> JudgeModel:
        """Get the judge model."""
        return self.judge_model

    def get_info(self) -> Dict[str, Any]:
        """Get metadata about the organization."""
        return {
            "name": "DeepSeek",
            "candidate_model": self.candidate_model.get_info(),
            "evaluator_model": self.evaluator_model.get_info(),
            "judge_model": f"{self.judge_provider}/{self.judge_model.model_name}",
        }


class MistralOrganization(Organization):
    """Mistral organization using Mistral 7B Instruct."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        judge_provider: str = "openai",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        judge_model_name: str = "gpt-4.1-2025-04-14",
    ):
        """
        Initialize the Mistral organization with Mistral 7B Instruct model.

        Args:
            api_key: Optional API key for Together AI.
            judge_provider: Provider for judge model ('openai', 'anthropic', or 'google').
            openai_api_key: Optional API key for OpenAI models.
            anthropic_api_key: Optional API key for Anthropic models.
            judge_model_name: Name of the judge model to use.
        """
        self.client = Together(api_key=api_key)
        self.candidate_model = TogetherCandidateModel(
            "mistralai/Mistral-7B-Instruct-v0.3", self.client
        )
        self.evaluator_model = OpenAIEvaluatorModel(api_key=openai_api_key)
        print(f"evaluator_model is : {self.evaluator_model.model_name}")

        self.judge_provider = judge_provider
        self.judge_model_name = judge_model_name
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key

        if judge_provider.lower() == "openai":
            self.judge_model = OpenAIJudgeModel(model_name=judge_model_name, api_key=openai_api_key)
        elif judge_provider.lower() == "anthropic":
            self.judge_model = AnthropicJudgeModel(
                model_name=judge_model_name, api_key=anthropic_api_key
            )
        elif judge_provider.lower() == "google":
            self.judge_model = GoogleJudgeModel(model_name=judge_model_name)
        else:
            self.judge_model = OpenAIJudgeModel(api_key=openai_api_key)

        print(f"judge_model is : {self.judge_model.model_name}")

    def get_candidate_model(self) -> CandidateModel:
        """Get the candidate model."""
        return self.candidate_model

    def get_evaluator_model(self) -> EvaluatorModel:
        """Get the evaluator model."""
        return self.evaluator_model

    def get_judge_model(self) -> JudgeModel:
        """Get the judge model."""
        return self.judge_model

    def get_info(self) -> Dict[str, Any]:
        """Get metadata about the organization."""
        return {
            "name": "Mistral",
            "candidate_model": self.candidate_model.get_info(),
            "evaluator_model": self.evaluator_model.get_info(),
            "judge_model": f"{self.judge_provider}/{self.judge_model.model_name}",
        }
