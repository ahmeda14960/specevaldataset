#!/usr/bin/env python
"""
Fix YAML formatting issues in Anthropic spec files
"""
import os
import glob
import re
import yaml # Import yaml to test parsing

def fix_yaml_file(file_path):
    """Fix formatting issues in a YAML file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        original_content = content # Keep original for comparison

        # --- Refined Fixes (Version 2 - worked for most files) ---

        # 1. Normalize indentation after block scalar indicator (|)
        content = re.sub(r'(: \\|\\n)\\s*(\\S)', r'\\1        \\2', content) # Text starts right after '|\\n'
        content = re.sub(r'(: \\|\\n)\\s*\\n\\s*(\\S)', r'\\1\\n        \\2', content) # Blank line then text

        # 2. Remove quotes wrapping the entire block scalar content
        content = re.sub(r'(: \\|\\n\\s*)\"(.*?)\"(\\s*)$\', r'\\1\\2\\3\', content, flags=re.DOTALL)

        # --- End Refined Fixes ---

        # Attempt to parse the modified content to check validity
        parse_success = False # Default to false
        if content != original_content:
            try:
                yaml.safe_load(content)
                parse_success = True
            except yaml.YAMLError as e:
                print(f"Warning: Failed to parse {os.path.basename(file_path)} after fixing attempt (reverting): {e}")
                content = original_content
                parse_success = False
        else:
             try:
                yaml.safe_load(content)
                parse_success = True
             except yaml.YAMLError:
                 # This case means original was invalid, and script didn't change it.
                 # We will report this as failure later.
                 parse_success = False

        # Write back the fixed content ONLY if it changed AND parses successfully
        if content != original_content and parse_success:
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Fixed: {os.path.basename(file_path)}")
            return True # Indicates a successful fix was applied
        elif content == original_content and parse_success:
            # print(f"No changes needed (already valid): {os.path.basename(file_path)}")
            return True # Still considered "success" in terms of processing
        else: # Covers (original invalid, no changes) and (changed, parse failed)
            if content != original_content: # Parsing failed after change
                 print(f"Failed (parsing error after change): {os.path.basename(file_path)}")
            else: # Original invalid, no change made
                 print(f"Skipped (invalid, no changes made): {os.path.basename(file_path)}")
            return False # Indicates failure

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return False

def main():
    """Process all YAML files in the anthropic directory"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_dir = os.path.join(script_dir, '..', 'data', 'specs', 'anthropic')
    yaml_files = glob.glob(os.path.join(yaml_dir, '*.yaml'))

    fixed_count = 0
    failed_count = 0
    total_files = len(yaml_files)

    # Process each file
    for file_path in yaml_files:
        if fix_yaml_file(file_path):
            fixed_count += 1
        else:
            failed_count += 1

    print(f"\nProcessing complete.")
    print(f"Successfully processed/fixed: {fixed_count}/{total_files}")
    if failed_count > 0:
        print(f"Failed to fix/parse: {failed_count}/{total_files}")

if __name__ == "__main__":
    main()
