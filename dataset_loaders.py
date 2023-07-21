import torch
from torch.utils import data

class MaterialSynthesisDataset(data.Dataset):
    """
    Produces tokenized pairs of sentences and labels.
    """
    def __init__(self, sentences, labels, label_dict, tokenizer, max_input_length):
            assert(len(sentences)==len(labels))
            self.sentences = sentences
            self.labels = labels
            self.label_dict = label_dict
            self.size = len(self.sentences)

            self.tokenizer = tokenizer
            self.max_input_length = max_input_length
            self.data = { "input_ids": [], "attention_mask": [], "labels": [] }
            self.tokenize_and_add_data(self.sentences, self.labels)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.data.items()}
        return item

    def __len__(self):
        return self.size
    
    def align_labels(self, tokens, labels):
        aligned_labels = []
        for word_idx in tokens.word_ids():
            # Special tokens have a word id that is None. 
            # We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                aligned_labels.append(-100)
            else:
                aligned_labels.append(self.label_dict[labels[word_idx]])

        return aligned_labels

    def tokenize_and_add_data(self, sentences, labels):
        for i, sentence in enumerate(sentences):
            tokens = self.tokenizer(
                sentence,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=self.max_input_length,
                is_split_into_words=True)
            self.data["input_ids"].append(torch.reshape(tokens["input_ids"], (-1,)))
            self.data["attention_mask"].append(torch.reshape(tokens["attention_mask"], (-1,)))
            self.data["labels"].append(self.align_labels(tokens, labels[i]))
    