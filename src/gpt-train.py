import random
from typing import Any

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer, max_len):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for data in data_list:
            data_dict = tokenizer('<|startoftext|>' + data + '<|endoftext|>', truncation=True, max_length=max_len,
                                  padding='max_length')
            self.input_ids.append(torch.tensor(data_dict['input_ids']))
            self.attn_masks.append(torch.tensor(data_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


def train(dataset, model, tokenizer) -> None:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # def compute_metrics(eval_preds):
    #     metric = load_metric("bleu")
    #     # logits, labels = eval_preds
    #     # predictions = np.argmax(logits, axis=-1)
    #     return metric.compute(predictions=eval_preds, smooth_method='floor')

    args = TrainingArguments(
        output_dir='./model/trek_summary_gpt2',
        num_train_epochs=5,
        learning_rate=0.005,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=args,
        data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                    'attention_mask': torch.stack([f[1] for f in data]),
                                    # 'labels': torch.stack([f[0] for f in data])}
                                    }
    )

    trainer.train()
    trainer.save_model()
    trainer.model.generate()


def make_dataset(filename: str, tokenizer: GPT2Tokenizer) -> Any:
    data_list = []
    total_len = 0
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            data_list.append(line.strip())
            total_len += len(line.strip())
    max_len = int(total_len/len(data_list)) + 5  # making the max length a bit longer than average because that seems right
    random.shuffle(data_list)
    data_dict = tokenizer('<|startoftext|>' + data_list[0] + '<|endoftext|>', truncation=True, max_length=max_len,
                          padding='max_length')

    print(data_dict)
    print(tokenizer.decode(data_dict['input_ids']))
    dataset = Dataset(data_list, tokenizer, max_len)
    # train_dataset = dataset[:int(0.9 * len(dataset))]
    # val_dataset = dataset[int(0.9 * len(dataset)):]
    # print(ty)
    # print(tokenizer.decode(dataset[:3]))
    return dataset


def main():

    # def tokenize(batch) -> Any:
    #     inputs = tokenizer(batch['Summaries'], padding=True)
    #     return inputs

    filename = './data/star_trek_episode_summaries.csv'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '<|pad|>',
                                  'bos_token': '<|startoftext|>',
                                  'eos_token': '<|endoftext|>'})

    train_dataset= make_dataset(filename, tokenizer)

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    #
    # summary_data = load_dataset('csv', data_files=filename, ).shuffle(seed=42)
    # summary_split = summary_data['train'].train_test_split(train_size=0.8, seed=42)
    # train_dataset = summary_split['train'].map(tokenize, batched=True)
    # val_dataset = summary_split['test'].map(tokenize, batched=True)

    train(train_dataset, model, tokenizer)


if __name__ == '__main__':
    main()
