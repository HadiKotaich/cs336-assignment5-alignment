import json
import os
from unittest.mock import patch

import torch
import wandb
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from scripts.evaluate_baseline import evaluate_vllm
from scripts.load_sft_data import get_sft_dataloader
from scripts.sft_experiment import init_vllm, load_policy_into_vllm_instance, run_sft
from tests import adapters
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from vllm.entrypoints.llm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

EVAL_DIR = (
    "/data/users/hadikotaich/cs336-assignment5-alignment/eval_results/expert_iteration"
)


class EISFTDataset(Dataset):
    """PyTorch Dataset for SFT training data from MetaMathQA."""

    def __init__(self, data: list[dict[str, str]]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.data[idx]


def run_expert_iteration() -> None:
    training_data_size = 8192
    expert_iteration_batch = 1024
    sft_batch_size = 8
    learning_rate = 0.0001
    rollouts = 8
    run_id = (
        f"ei_sz{training_data_size}_lr_{learning_rate}_eibs_{expert_iteration_batch}_sftbs_{sft_batch_size}"
        + wandb.util.generate_id()
    )
    wandb.init(project="ei", id=run_id)
    # model_id = "/data/users/hadikotaich/cs336-assignment5-alignment/models/Qwen2.5-Math-1.5B-sft-sft_sz512_lr_0.0001_bs_10_4xp44m6o"
    model_id = (
        "/data/users/hadikotaich/cs336-assignment5-alignment/models/Qwen2.5-Math-1.5B"
    )
    model_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(model_device)

    data_loader = get_sft_dataloader(
        num_rows=training_data_size, batch_size=expert_iteration_batch, shuffle=False
    )
    vllm = init_vllm(
        model_id=model_id,
        seed=42,
        gpu_memory_utilization=0.75,
        tensor_parallel_size=1,
    )
    load_policy_into_vllm_instance(model, vllm)
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    evaluate_vllm(
        vllm,
        output_path=f"{EVAL_DIR}/expert_iteration/before_training_eval.jsonl",
    )

    all_results = []

    for idx, batch in enumerate(data_loader):
        # Collect all results for analysis
        batch_results = []
        print(f"starting EI batch with index {idx}")
        ids = batch["id"]
        prompts = batch["prompt"]
        ground_truths = batch["response"]
        # generate model output
        print(f"generating outputs for batch with index {idx}")
        for _ in range(rollouts):
            outputs: list[RequestOutput] = vllm.generate(prompts, sampling_params)
            sft_data = []
            for rowd_id, prompt, output, ground_truth in zip(
                ids, prompts, outputs, ground_truths
            ):
                # Extract ground truth answer from between <answer> and </answer> tags
                if "<answer>" in ground_truth and "</answer>" in ground_truth:
                    ground_truth = (
                        ground_truth.split("<answer>")[-1].split("</answer>")[0].strip()
                    )
                output_text = output.outputs[0].text
                metrics = r1_zero_reward_fn(output_text, ground_truth)

                # Save all results for analysis
                batch_results.append(
                    {
                        "id": rowd_id,
                        "prompt": prompt,
                        "output": output_text,
                        "ground_truth": ground_truth,
                        "metrics": metrics,
                    }
                )

                # print(f"{output_text=}")
                # print(f"{ground_truth=}")
                # print(f"{metrics=}")
                reward = metrics["reward"]
                if reward > 0:
                    # found good training data
                    sft_data.append(
                        {
                            "id": rowd_id,
                            "prompt": prompt,
                            "response": output_text,
                        }
                    )
        # Print summary statistics
        all_results.extend(batch_results)
        total = len(batch_results)
        correct_format = sum(
            1 for r in batch_results if r["metrics"]["format_reward"] > 0
        )
        correct_answer = sum(
            1 for r in batch_results if r["metrics"]["answer_reward"] > 0
        )
        total_reward = sum(1 for r in batch_results if r["metrics"]["reward"] > 0)

        print(f"\nSummary Statistics:")
        print(f"Total examples: {total}")
        print(f"Correct format: {correct_format} ({100*correct_format/total:.2f}%)")
        print(f"Correct answer: {correct_answer} ({100*correct_answer/total:.2f}%)")
        print(f"Total reward > 0: {total_reward} ({100*total_reward/total:.2f}%)")

        if len(sft_data) == 0:
            print(f"no good training data found for {idx=}, skipping sft")
            continue

        sft_data_loader = DataLoader(
            EISFTDataset(sft_data), batch_size=sft_batch_size, shuffle=False
        )
        run_sft(
            model=model,
            model_device=model_device,
            data_loader=sft_data_loader,
            model_id=model_id,
            run_id=run_id,
            learning_rate=learning_rate,
            eval_frequency=0,
        )

        print(f"done with sft, now evaluating...")

        load_policy_into_vllm_instance(model, vllm)
        evaluate_vllm(
            vllm,
            output_path=f"{EVAL_DIR}/expert_iteration/{idx}_eval.jsonl",
        )

    # Save all results to a file
    output_file = f"{EVAL_DIR}/expert_iteration_results_{idx}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {output_file}")


if __name__ == "__main__":
    run_expert_iteration()
