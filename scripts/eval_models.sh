
python -m examples.eval_openai_model_spec --verbose --test-all-statements --num-inputs-per-statement 20
python -m examples.eval_anthropic_models --test-all-statements --num-inputs-per-statement 20 --judge-provider openai --evaluator-provider openai --judge-model gpt-4o-2024-08-06  --evaluator-model gpt-4o-2024-08-06
python -m examples.eval_together_models --test-all-statements --num-inputs-per-statement 20
