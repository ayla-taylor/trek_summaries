from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()

model_dir = '../model/StarTrekSummary_Model'
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)


def generate():
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
    for sample_output in sample_outputs:
        yield 'tokenizer.decode(sample_output, skip_special_tokens=True)}'


@app.get('/predict')
def main():
    text = str(generate())
    return {'message': text}


@app.get('/{name}')
def hello_name(name: str):
    return {'message': f'Blarg blarg {name}'}