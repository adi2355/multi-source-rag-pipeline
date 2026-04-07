#!/usr/bin/env python3
"""
Script to fix indentation issues in github_collector.py
"""

import re
import sys

def fix_indentation(file_path):
    """
    Fix indentation issues in the Python file.
    
    Args:
        file_path (str): Path to the file to fix
    """
    print(f"Fixing indentation in {file_path}")
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Process lines
    fixed_lines = []
    in_function = False
    current_indentation = 0
    
    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            fixed_lines.append(line)
            continue
        
        # Check if line defines a function or class
        if re.match(r'^def\s+|^class\s+', line):
            in_function = True
            current_indentation = 0
            fixed_lines.append(line)
            continue
        
        # Handle indentation
        stripped_line = line.lstrip()
        if not stripped_line:
            fixed_lines.append(line)
            continue
        
        indent = len(line) - len(stripped_line)
        
        # Simple heuristic: content after function definition should have at least 4 spaces
        if in_function and indent > 0 and indent < 4:
            # Fix lines with incorrect indentation
            fixed_line = ' ' * 4 + stripped_line
            fixed_lines.append(fixed_line)
            print(f"Fixed line {i+1}: {line.rstrip()} -> {fixed_line.rstrip()}")
        else:
            fixed_lines.append(line)
    
    # Write the fixed content back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"Indentation fixes applied to {file_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_indentation.py <file_path>")
        sys.exit(1)
    
    fix_indentation(sys.argv[1])
    print("Done!") 