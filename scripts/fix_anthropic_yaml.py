#!/usr/bin/env python3
# Script to fix YAML formatting issues in Anthropic specification files
import os
import glob
import re


def fix_file(path):
    print(f"Processing {os.path.basename(path)}")
    # Read the file content
    with open(path, "r") as f:
        content = f.read()

    # Fix the double pipe issue | |
    content = re.sub(r"(\s+good_response: )\| \|", r"\1|", content)
    content = re.sub(r"(\s+bad_response: )\| \|", r"\1|", content)

    # Fix responses that have pipe character followed by quotes - remove quotes
    content = re.sub(
        r'(\s+good_response: \|)\s*\n\s*"(.+?)"\s*(\n\s+(?:bad_response|description|user_query|-))',
        lambda m: f"{m.group(1)}\n{' ' * 8}{m.group(2)}{m.group(3)}",
        content,
        flags=re.DOTALL,
    )

    content = re.sub(
        r'(\s+bad_response: \|)\s*\n\s*"(.+?)"\s*(\n\s+(?:description|user_query|-|\Z))',
        lambda m: f"{m.group(1)}\n{' ' * 8}{m.group(2)}{m.group(3)}",
        content,
        flags=re.DOTALL,
    )

    # Fix responses that don't have pipe character
    content = re.sub(r"(\s+good_response: )([^|\n])", r"\1|\n        \2", content)
    content = re.sub(r"(\s+bad_response: )([^|\n])", r"\1|\n        \2", content)

    # Write back the changes
    with open(path, "w") as f:
        f.write(content)

    print(f"Fixed: {os.path.basename(path)}")
    return True


def main():
    basedir = os.path.join(os.path.dirname(__file__), "..", "data", "specs", "anthropic")
    pattern = os.path.join(basedir, "*.yaml")
    fixed_count = 0

    for path in glob.glob(pattern):
        try:
            if fix_file(path):
                fixed_count += 1
        except Exception as e:
            print(f"Error processing {os.path.basename(path)}: {e}")

    print(f"Fixed {fixed_count} files")


if __name__ == "__main__":
    main()
