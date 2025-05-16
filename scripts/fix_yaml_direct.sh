#!/bin/bash
# Direct YAML file fixer using sed

YAML_DIR="$(dirname "$(dirname "$0")")/data/specs/anthropic"
echo "Working in directory: $YAML_DIR"

# Make a backup directory
BACKUP_DIR="$YAML_DIR/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Backing up files to $BACKUP_DIR"

# Copy all yaml files to backup
cp "$YAML_DIR"/*.yaml "$BACKUP_DIR"

# Process each YAML file
for file in "$YAML_DIR"/*.yaml; do
    filename=$(basename "$file")
    echo "Processing $filename"

    # Make a temporary file
    temp_file=$(mktemp)

    # Remove quotes from block scalars and fix indentation
    # 1. First make sure the right indentation is there for block scalars
    # 2. Then remove quotes around text content
    # Use -E for extended regular expressions
    sed -E '
        # Fix indentation for block scalar following pipe
        s/(good_response|bad_response): \|$/\1: |/g

        # Fix quotes around block scalar content - remove the quotes and add proper indentation
        s/^(\s+)(good_response|bad_response): \|\s*$/\1\2: |/g
        s/^(\s+)"(.+)"$/\1\2/g

        # Add proper indentation after pipe symbol
        s/^(\s+)(good_response|bad_response): \|$/\1\2: |\n\1        /g
    ' "$file" > "$temp_file"

    # Replace original with fixed version
    mv "$temp_file" "$file"
done

echo "Finished processing all files"
