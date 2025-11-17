import json
import os
import time
from itertools import batched
from pathlib import Path
from typing import Callable, List

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from datasets import DatasetDict, load_dataset

from datasets.features.features import Sequence
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

# RUN:
# python scripts/evaluate_baseline.py

# DESCRIPTION:
# load math validation examples
# format them as prompts to llm
# generate outputs from llm
# calculate evaluation metrics
# serialize examples, model generations, evaluation scores to disk


class DataRow(BaseModel):
    question: str
    answer: str


class EvalRow(BaseModel):
    prompt: str
    response: str
    ground_truth: str
    metrics: dict[str, float]


def get_gsm8k_data(split) -> list[DataRow]:
    dataset: DatasetDict = load_dataset("openai/gsm8k", "main", split=split)
    return [DataRow(question=row["question"], answer=row["answer"]) for row in dataset]


def get_prompts(questions: list[str]) -> list[str]:
    prompt_path = (
        Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
    )
    with open(prompt_path) as f:
        prompt = f.read()
    return [prompt.format(question=q) for q in questions]


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    batch_size = 1024
    format_reward = 0.0
    answer_reward = 0.0
    reward = 0.0
    start_time = time.time()
    outputs: list[RequestOutput] = vllm_model.generate(prompts, eval_sampling_params)
    generation_latency = time.time() - start_time

    start_time = time.time()
    with open("eval_results/baseline_metrics.jsonl", "w") as fout:
        for prompt, ground_truth, output in zip(prompts, ground_truths, outputs):
            response = output.outputs[0].text

            eval_row = EvalRow(
                prompt=prompt,
                response=response,
                ground_truth=ground_truth,
                metrics=reward_fn(response, ground_truth),
            )

            format_reward += eval_row.metrics["format_reward"]
            answer_reward += eval_row.metrics["answer_reward"]
            reward += eval_row.metrics["reward"]
            fout.write(json.dumps(eval_row.model_dump()) + "\n")
    eval_latency = time.time() - start_time

    print(f"\nLatency Statistics:")
    print(f"  Generation latency: {generation_latency:.2f} seconds")
    print(f"  Evaluation latency: {eval_latency:.2f} seconds")

    print(f"\nReward Statistics:")
    print(f"  Format reward: {format_reward / len(prompts):.2f}")
    print(f"  Answer reward: {answer_reward / len(prompts):.2f}")
    print(f"  Reward: {reward / len(prompts):.2f}")


def extract_gsm8k_answer(gsm8k_answer: str) -> str:
    if "####" in gsm8k_answer:
        return gsm8k_answer.split("####")[-1].strip()
    return gsm8k_answer


def main():
    # qwen3_1_7b = LLM(model="Qwen/Qwen3-1.7B")
    model = LLM(model="Qwen/Qwen2.5-1.5B")
    data_rows: list[DataRow] = get_gsm8k_data(split="test")
    prompts: list[str] = get_prompts([row.question for row in data_rows])
    ground_truths: list[str] = [extract_gsm8k_answer(row.answer) for row in data_rows]
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    evaluate_vllm(
        vllm_model=model,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
    )


if __name__ == "__main__":
    main()
