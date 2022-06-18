from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_dir = '../model/StarTrekSummary_Model'
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

model.eval()

prompt = "<bos>"

generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

sample_outputs = model.generate(
                                generated,
                                do_sample=True,
                                top_k=50,
                                max_length=100,
                                top_p=0.95,
                                num_return_sequences=3
                            )

for i, sample_output in enumerate(sample_outputs):
    print(f'{i}: {tokenizer.decode(sample_output, skip_special_tokens=True)}\n')