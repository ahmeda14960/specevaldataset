"""Together model implementations for the SpecEval framework."""
from typing import Dict, Any
from together import Together
import time

from ..base import CandidateModel

# Model name constants
# Deepseek
DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"

# Qwen
QWEN_235B_FP8 = "Qwen/Qwen3-235B-A22B-fp8-tput"
QWEN_2_5_72B_TURBO = "Qwen/Qwen2.5-72B-Instruct-Turbo"
QWEN_2_72B_INSTRUCT = "Qwen/Qwen2-72B-Instruct"

# Meta
LLAMA_4_MAVERICK_17B = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
LLAMA_3_1_405B_TURBO = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"


class TogetherCandidateModel(CandidateModel):
    """Together AI candidate model implementation."""

    def __init__(self, model_name: str, client: Together):
        """
        Initialize the Together AI candidate model.

        Args:
            model_name: Name of the Together AI model to use.
            client: Together AI client instance.
        """
        self.model_name = model_name
        self.client = client

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response using the Together AI API."""
        try:
            # Add rate limiting for deepseek models
            if "deepseek" in self.model_name.lower():
                time.sleep(10)  # Wait for 10 seconds

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Error generating response with model {self.model_name}: {e}")

    def get_info(self) -> Dict[str, Any]:
        """Return metadata about the model."""
        return {
            "model_name": self.model_name,
            "provider": "together",
        }

    def supports_caching(self) -> bool:
        """Together models support caching."""
        return True
