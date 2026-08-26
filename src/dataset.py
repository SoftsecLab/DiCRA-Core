import json
import torch
from torch.utils.data import Dataset

class JSONLDataset(Dataset):
    def __init__(self, path, tokenizer=None, max_len=512, encode_on_getitem=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.encode_on_getitem = encode_on_getitem
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if not self.encode_on_getitem:
            return {
                'text': item['text'],
                'label': int(item['label'])
            }

        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided when encode_on_getitem=True")

        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(item['label']), dtype=torch.long)
        }


class JSONLBatchCollator:
    def __init__(self, tokenizer, max_len=512, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch):
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt'
        )
        encoding['labels'] = labels
        return encoding
