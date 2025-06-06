#!/usr/bin/env python3
import re
import sys
from collections import defaultdict

def extract_suppressions(file_path):
    """
    Extract suppression lists from Helgrind output, count occurrences, and print each suppression with its count.

    :param file_path: Path to the file containing Helgrind output
    """
    # Regular expression to match suppression blocks
    suppression_start_pattern = r"^{"
    suppression_end_pattern = r"^}$"

    # Dictionary to store suppression blocks and their counts
    suppressions = defaultdict(int)

    with open(file_path, 'r') as file:
        lines = file.readlines()

    inside_suppression = False
    current_suppression = []

    for line in lines:
        line = line.rstrip()  # Remove trailing whitespace

        if re.match(suppression_start_pattern, line):
            inside_suppression = True
            current_suppression = [line]
        elif inside_suppression:
            current_suppression.append(line)
            if re.match(suppression_end_pattern, line):
                inside_suppression = False
                # Join the suppression block into a single string
                suppression_block = "\n".join(current_suppression)
                # Increment the count for this suppression block
                suppressions[suppression_block] += 1

    # Print each unique suppression block with its count
    print("Unique Suppressions and Counts:")
    for suppression, count in suppressions.items():
        print(f"\nCount: {count}\nSuppression:\n{suppression}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./extract_suppressions.py <helgrind_output_file>")
        print("Extracts and counts unique suppression lists from Helgrind output.")
        sys.exit(1)

    helgrind_output_file = sys.argv[1]
    extract_suppressions(helgrind_output_file)