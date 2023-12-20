import os
import csv

file_path = os.path.join(os.path.dirname(__file__), 'law_related_questions.csv')

# Read existing content
existing_content = []
with open(file_path, 'r') as f:
    reader = csv.reader(f)
    existing_content = list(reader)

# Find the last entry that ends with a comma
last_entry_index = len(existing_content) - 1

# Ensure the list is not empty before checking indices
if last_entry_index >= 0:
    while last_entry_index >= 0:
        if existing_content[last_entry_index] and existing_content[last_entry_index][-1].endswith(''):
            break
        last_entry_index -= 1

    # Modify the existing content
    if last_entry_index >= 0:
        existing_content[last_entry_index].append('i love you')

        # Write the modified content back to the file
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(existing_content)

