import json
import random

def generate_instruction_response_pairs_with_blanks_everywhere(texts):
    data = []
    for text in texts:
        words = text.split()
        for i in range(len(words)):
            modified_words = words.copy()
            response = modified_words[i] + '.'  # Capture the response before replacing with a blank
            modified_words[i] = "_"  # Replace the current word with a blank
            instruction = 'What should be in the blank: ' + ' '.join(modified_words) + "?"
            pair = {
                "instruction": instruction,
                "context": "",
                "response": response,
                "category": "brainstorming"
            }
            data.append(pair)
    return data

# Example sentences
texts = [
    "The quick brown fox jumps over the lazy dog",
    "An apple a day keeps the doctor away"
]

# Generate pairs
pairs = generate_instruction_response_pairs_with_blanks_everywhere(texts)

# Write to JSON file
with open('instruction_response_pairs_everywhere.json', 'w') as file:
    json.dump(pairs, file, indent=4)
