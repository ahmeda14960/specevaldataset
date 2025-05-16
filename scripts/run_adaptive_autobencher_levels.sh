#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Define common arguments
ROUNDS=5
SELECTION_PERCENT=0.33
INPUTS_PER_SCENARIO=12
NUM_STATEMENTS=50
VERBOSE="--verbose"
BASE_CMD="python -m examples.adaptive_autobencher \
    --rounds $ROUNDS \
    --selection-percent $SELECTION_PERCENT \
    --inputs-per-scenario $INPUTS_PER_SCENARIO \
    --num-statements $NUM_STATEMENTS \
    $VERBOSE"

# Run for 'user' authority level
echo "Running Adaptive AutoBencher for authority level: user"
$BASE_CMD --authority-level user
echo "Finished running for authority level: user"

echo "\n----------------------------------------\n"

# Run for 'organization' authority level
echo "Running Adaptive AutoBencher for authority level: organization"
$BASE_CMD --authority-level organization
echo "Finished running for authority level: organization"

echo "\nScript completed successfully."
