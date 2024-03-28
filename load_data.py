import torch
from torch.utils.data import Dataset, DataLoader
import json

class ShakespeareDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.samples = []
        self.users = {}
        for user in self.data['users']:
            user_samples = self.data['user_data'][user]['x']
            user_answers = self.data['user_data'][user]['y']
            user_sample = []
            for x, y in zip(user_samples, user_answers):
                user_sample.append((x, y))
            self.users[user] = user_sample
            self.samples.extend(user_sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_x, sample_y = self.samples[idx]
        return sample_x, sample_y
    
    def generate_instruction_response_pairs_with_blanks_everywhere(self, user=None):
        data = []
        if user == None:
            origin_data = self.samples
        else:
            origin_data = self.users[user]
        for sample_x, sample_y in origin_data:
            instruction = 'What letter should be after this:\"' + sample_x + '\"?'
            pair = {
                "instruction": instruction,
                "context": "",
                "response": sample_y,
                "category": "brainstorming"
            }
            data.append(pair)
        return data

# Training data
# Usage
training_json_path = './shakespeare/data/train/all_data_niid_0_keep_0_train_9.json'
training_dataset = ShakespeareDataset(training_json_path)

# Generate instruction-response pairs
for user in training_dataset.users.keys():
    pairs = training_dataset.generate_instruction_response_pairs_with_blanks_everywhere(user)
    output_json_path = './data/training/user/shakespeare_instruction_response_pairs_'+user+'.json'
    with open(output_json_path, 'w') as file:
        json.dump(pairs, file, indent=4)
pairs = training_dataset.generate_instruction_response_pairs_with_blanks_everywhere()
output_json_path = './data/training/shakespeare_instruction_response_pairs_all.json'
with open(output_json_path, 'w') as file:
        json.dump(pairs, file, indent=4)




# Testinging data
# Usage
testing_json_path = './shakespeare/data/test/all_data_niid_0_keep_0_test_9.json'
testing_dataset = ShakespeareDataset(testing_json_path)

# Generate instruction-response pairs
for user in testing_dataset.users.keys():
    pairs = testing_dataset.generate_instruction_response_pairs_with_blanks_everywhere(user)
    output_json_path = './data/testing/user/shakespeare_instruction_response_pairs_'+user+'.json'
    with open(output_json_path, 'w') as file:
        json.dump(pairs, file, indent=4)
pairs = testing_dataset.generate_instruction_response_pairs_with_blanks_everywhere()
output_json_path = './data/testing/shakespeare_instruction_response_pairs_all.json'
with open(output_json_path, 'w') as file:
        json.dump(pairs, file, indent=4)
