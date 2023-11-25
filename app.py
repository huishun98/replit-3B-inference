import os
from dataclasses import dataclass, asdict
from ctransformers import AutoModelForCausalLM, AutoConfig
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: list[str]


class CompletionRequest(BaseModel):
    text: str


config = AutoConfig.from_pretrained(
    os.path.abspath("models"),
    context_length=2048,
)
llm = AutoModelForCausalLM.from_pretrained(
    os.path.abspath("models/replit-v2-codeinstruct-3b.q4_1.bin"),
    model_type="replit",
    config=config,
)

generation_config = GenerationConfig(
    temperature=0.2,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0,
    max_new_tokens=512,  # adjust as needed
    seed=42,
    reset=True,  # reset history (cache)
    stream=True,  # streaming per word/token
    threads=int(os.cpu_count() / 6),  # adjust for your CPU
    stop=["<|endoftext|>"],
)


@app.get("/")
async def root():
    return {"hello": "world"}


@app.post("/")
def generate(body: CompletionRequest):
    """run model inference, will return a Generator if streaming is true"""
    print(body.text)
    resp = llm(body.text, **asdict(generation_config))
    print(resp)
    return {'resp': resp}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
