#!/usr/bin/env python3
"""
Simple interactive script to test the base CandidateModel from a chosen organization.

Allows sending prompts to the model via the terminal and receiving responses.

Example Usage:

# Test OpenAI (ensure OPENAI_API_KEY is set)
python examples/test_model_org.py --org OpenAI

# Test Google (ensure GOOGLE_API_KEY is set)
from the gemini docs:
# Replace the `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` values
# with appropriate values for your project.
export GOOGLE_CLOUD_PROJECT=hai-gcp-models
export GOOGLE_CLOUD_LOCATION=us-central2
export GOOGLE_GENAI_USE_VERTEXAI=True

python examples/test_model_org.py --org Google --candidate-model gemini-2.0-flash-001

# Test Anthropic (ensure ANTHROPIC_API_KEY is set)
python examples/test_model_org.py --org Anthropic

# Test Meta models (ensure TOGETHER_API_KEY is set)
python examples/test_model_org.py --org MetaOrganization --candidate-model meta-llama/Llama-2-70b-chat-hf

# Test Qwen models
python examples/test_model_org.py --org QwenOrganization --candidate-model Qwen/Qwen-14B-Chat

# Test DeepSeek models
python examples/test_model_org.py --org DeepSeekOrganization --candidate-model deepseek-ai/deepseek-coder-33b-instruct

# Provide API key via argument (not recommended for security)
# python examples/test_model_org.py --org OpenAI --openai-api-key YOUR_KEY_HERE
"""

import os
import argparse
import sys


from speceval.orgs import (
    OpenAI,
    Anthropic,
    Google,
    MetaOrganization,
    QwenOrganization,
    DeepSeekOrganization,
    MixedOrganization,
    # Add other organizations from your __init__.py if needed
)
from speceval.base import Organization, CandidateModel


# Define available organizations based on imports
AVAILABLE_ORGS = {
    "OpenAI": OpenAI,
    "Anthropic": Anthropic,
    "Google": Google,
    "MetaOrganization": MetaOrganization,
    "QwenOrganization": QwenOrganization,
    "DeepSeekOrganization": DeepSeekOrganization,
    "MixedOrganization": MixedOrganization,
    # Add other org classes here if they are in __all__
}


def main():
    parser = argparse.ArgumentParser(
        description="Interact with a CandidateModel from a specified organization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--org",
        type=str,
        required=True,
        choices=list(AVAILABLE_ORGS.keys()),
        help="The organization whose candidate model you want to test.",
    )
    # Optional: Specify a particular model name for the candidate
    parser.add_argument(
        "--candidate-model",
        type=str,
        default=None,
        help="(Optional) Specific candidate model name to use within the organization.",
    )
    # API Key Arguments (Optional - Environment variables preferred)
    parser.add_argument(
        "--openai-api-key", type=str, default=None, help="OpenAI API key (env: OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--anthropic-api-key",
        type=str,
        default=None,
        help="Anthropic API key (env: ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--google-api-key", type=str, default=None, help="Google API key (env: GOOGLE_API_KEY)"
    )
    parser.add_argument(
        "--together-api-key",
        type=str,
        default=None,
        help="Together API key (env: TOGETHER_API_KEY)",
    )
    # Add arguments for other keys if needed (e.g., for MixedOrganization)

    args = parser.parse_args()

    # --- API Key Resolution ---
    openai_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    anthropic_key = args.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    google_key = args.google_api_key or os.environ.get("GOOGLE_API_KEY")
    together_key = args.together_api_key or os.environ.get("TOGETHER_API_KEY")

    # --- Organization Instantiation ---
    org_class = AVAILABLE_ORGS.get(args.org)
    if not org_class:
        # This should not happen due to choices constraint, but good practice
        print(f"Error: Unknown organization '{args.org}'")
        sys.exit(1)

    print(f"Initializing organization: {args.org}...")
    organization: Organization

    try:
        # Pass relevant keys and model name to the constructor
        # Note: Each org might have different constructor arguments
        org_kwargs = {}
        if args.candidate_model:
            # Most orgs expect candidate_model_name
            org_kwargs["candidate_model_name"] = args.candidate_model

        if args.org == "OpenAI":
            if not openai_key:
                raise ValueError("OpenAI API key is required.")
            org_kwargs["api_key"] = openai_key
            organization = OpenAI(**org_kwargs)
        elif args.org == "Anthropic":
            if not anthropic_key:
                raise ValueError("Anthropic API key is required.")
            org_kwargs["api_key"] = anthropic_key
            organization = Anthropic(**org_kwargs)
        elif args.org == "Google":
            org_kwargs["api_key"] = google_key
            organization = Google(**org_kwargs)
        elif args.org in ["MetaOrganization", "QwenOrganization", "DeepSeekOrganization"]:
            if not together_key:
                raise ValueError("Together API key is required.")
            # Together org might take the key differently, adjust if needed
            org_kwargs["api_key"] = together_key
            organization = org_class(**org_kwargs)
        elif args.org == "MixedOrganization":
            # MixedOrganization likely needs multiple keys and specific config
            print("Warning: MixedOrganization requires specific configuration.")
            print("Attempting basic initialization, may need more args/keys.")
            # Example: it might need all keys
            org_kwargs["openai_api_key"] = openai_key
            org_kwargs["anthropic_api_key"] = anthropic_key
            org_kwargs["google_api_key"] = google_key
            org_kwargs["together_api_key"] = together_key
            # Add other necessary keys/args for MixedOrganization
            organization = MixedOrganization(**org_kwargs)  # Ensure constructor matches
        else:
            # Fallback for any org added to AVAILABLE_ORGS but not handled above
            print(
                f"Warning: Specific initialization logic for {args.org} not implemented. Using default."
            )
            organization = org_class()  # Attempt default init

    except (ValueError, TypeError, KeyError) as e:
        print(f"Error initializing organization {args.org}: {e}")
        print("Please ensure the correct API key is provided via argument or environment variable")
        print("and that any required model names are specified if defaults are not suitable.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        sys.exit(1)

    # --- Get Candidate Model ---
    try:
        import ipdb

        ipdb.set_trace()
        candidate_model: CandidateModel = organization.get_candidate_model()
        model_info = candidate_model.get_info()
        print(
            f"Successfully loaded Candidate Model: {model_info.get('provider', '')}/{model_info.get('model_name', '')}"
        )
    except Exception as e:
        print(f"Error getting candidate model from {args.org}: {e}")
        sys.exit(1)

    # --- Interaction Loop ---
    print("\nEnter your prompts below. Type 'exit' or 'quit' to end the session.")
    print("---")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            if not user_input:
                continue

            print("Model: ...thinking...")
            # Use the generate method of the candidate model
            try:
                response = candidate_model.generate(user_input)
                print(f"Model: {response}")
            except Exception as e:
                print(f"\nError during generation: {e}\n")
                print("Please check the API key, model name, and network connection.")

        except EOFError:
            # Handle Ctrl+D
            break
        except KeyboardInterrupt:
            # Handle Ctrl+C
            break

    print("\nExiting chat.")


if __name__ == "__main__":
    main()
