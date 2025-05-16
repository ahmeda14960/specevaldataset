#!/usr/bin/env python
# Simple script to fix YAML files by removing problematic formatting
import os
import glob


def fix_file(file_path):
    """Fix a YAML file by removing problematic formatting"""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Track whether we're in a response block
    in_response = False
    line_with_pipe = False
    quotes_opened = False

    # Store fixed lines
    fixed_lines = []

    for line in lines:
        # Check if we're at a good_response or bad_response line
        if "good_response:" in line or "bad_response:" in line:
            in_response = True
            if "|" in line:
                line_with_pipe = True
            fixed_lines.append(line)
        elif in_response and line.strip() and not line.strip().startswith("-"):
            # We're inside a response block
            stripped_line = line.strip()

            # Check if this line starts with a quote
            if stripped_line.startswith('"') and not quotes_opened:
                quotes_opened = True
                stripped_line = stripped_line[1:]  # Remove opening quote

            # Check if this line ends with a quote
            if stripped_line.endswith('"') and quotes_opened:
                quotes_opened = False
                stripped_line = stripped_line[:-1]  # Remove closing quote

            # Add proper indentation
            fixed_line = "        " + stripped_line + "\n"
            fixed_lines.append(fixed_line)

            # Check if this is the end of the response
            if line.strip() == "":
                in_response = False
                line_with_pipe = False
                quotes_opened = False
        else:
            # Not in a response block, just add the line as is
            fixed_lines.append(line)
            if line.strip() == "":
                in_response = False
                line_with_pipe = False
                quotes_opened = False

    # Write the fixed content back to the file
    with open(file_path, "w") as f:
        f.writelines(fixed_lines)

    print(f"Fixed: {os.path.basename(file_path)}")
    return True


def main():
    """Fix all YAML files in the anthropic directory"""
    # Get the directory containing the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the anthropic specs directory
    yaml_dir = os.path.join(script_dir, "..", "data", "specs", "anthropic")

    # Find all YAML files
    yaml_files = glob.glob(os.path.join(yaml_dir, "*.yaml"))

    # Process each file
    count = 0
    for file_path in yaml_files:
        try:
            if fix_file(file_path):
                count += 1
        except Exception as e:
            print(f"Error with {os.path.basename(file_path)}: {e}")

    print(f"Fixed {count} out of {len(yaml_files)} files")


if __name__ == "__main__":
    main()
