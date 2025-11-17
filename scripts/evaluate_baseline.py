import os
import subprocess
from itertools import batched
from typing import Callable, List

from datasets.features.features import Sequence
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

# python scripts/evaluate_baseline.py

# load math validation examples

# format them as prompts to llm

# generate outputs from llm

# calculate evaluation metrics

# serialize examples, model generations, evaluation scores to disk


def get_free_gpu():
    # Query memory via nvidia-smi
    smi_query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    )
    memory_free = [int(x) for x in smi_query.decode("utf-8").strip().split("\n")]
    best_gpu = memory_free.index(max(memory_free))
    return best_gpu


gpu = get_free_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    batch_size = 32

    for batch in batched(prompts, 32):
        outputs: list[RequestOutput] = vllm_model.generate(batch, eval_sampling_params)
        for output in outputs:
            prompt = output.prompt
            response = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Response: {response!r}")


qwen3_1_7b = LLM(model="Qwen/Qwen3-1.7B", dtype="auto")
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)
evaluate_vllm(
    vllm_model=qwen3_1_7b,
    reward_fn=None,
    prompts=prompts,
    eval_sampling_params=sampling_params,
)
