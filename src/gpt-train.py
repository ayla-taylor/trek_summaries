from typing import Any

from transformers import GPT2Tokenizer, GPT2Model, Trainer, TrainingArguments
from datasets import load_metric


def train(dataset, model, tokenizer) -> None:
    metric = load_metric('bleu')


    args = TrainingArguments(
        output_dir='./model/trek_summary_gpt2',
        # evaluation_strategy='epoch',
        num_train_epochs=5,
        learning_rate=0.005
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=args,
        compute_metrics=metric

    )

    trainer.train()
    trainer.evaluate()


def tokenize_data(filename: str, tokenizer: GPT2Tokenizer) -> Any:
    summary_list = []
    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            summary_list.append(line.strip().strip(','))
    # average_len = int(sum([len(x) for x in summary_list])/len(summary_list))  # get average length of summary
    inputs = tokenizer(summary_list)
    return inputs


def main():
    filename = './data/star_trek_episode_summaries.csv'
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    dataset = tokenize_data(filename, tokenizer)

    train(dataset, model, tokenizer)


if __name__ == '__main__':
    main()
