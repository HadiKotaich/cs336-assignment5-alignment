# Patch importlib.metadata.distributions before wandb imports it
# to filter out packages with None metadata
# See: https://github.com/wandb/wandb/issues/10803
import copy
import importlib.metadata
from typing import Any, Callable, Literal

import torch
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from tests.adapters import (
    run_compute_group_normalized_rewards,
    run_grpo_microbatch_train_step,
)

_original_distributions = importlib.metadata.distributions


def _patched_distributions():
    """Filter out distributions with None metadata"""
    for dist in _original_distributions():
        if dist.metadata is not None:
            yield dist


importlib.metadata.distributions = _patched_distributions

import wandb
from scripts.evaluate_baseline import evaluate_vllm
from scripts.load_sft_data import get_sft_dataloader
from scripts.sft_experiment import init_vllm, load_policy_into_vllm_instance
from tests import adapters
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from vllm.entrypoints.llm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm.sampling_params import SamplingParams

EVAL_DIR = "/data/users/hadikotaich/cs336-assignment5-alignment/eval_results/grpo"


def grpo_train_loop(
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,  # As in Expiter, disallow empty string responses
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,  # On-policy
    train_batch_size: int = 256,  # On-policy
    gradient_accumulation_steps: int = 128,  # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85,
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline",
    use_std_normalization: bool = True,
    eval_frequency: int = 1,  # 0 means no eval
):
    run_id = f"grpo_{loss_type}_ngs_{n_grpo_steps}_lr_{learning_rate}_rbs_{rollout_batch_size}_gs_{group_size}_{wandb.util.generate_id()}"
    wandb.init(project="grpo_ass5", id=run_id)
    model_id = (
        "/data/users/hadikotaich/cs336-assignment5-alignment/models/Qwen2.5-Math-1.5B"
    )
    model_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    policy = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(model_device)

    data_loader = get_sft_dataloader(
        num_rows=train_batch_size * n_grpo_steps,
        batch_size=train_batch_size,
        shuffle=False,
    )
    vllm = init_vllm(
        model_id=model_id,
        seed=42,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
    )
    load_policy_into_vllm_instance(policy, vllm)

    # evaluate_vllm(
    #     vllm,
    #     output_path=f"{EVAL_DIR}/{run_id}/before_training_eval.jsonl",
    # )

    sampling_params = SamplingParams(
        temperature=sampling_temperature,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        top_p=1.0,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
    )
    assert (
        train_batch_size % gradient_accumulation_steps == 0
    ), "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert (
        rollout_batch_size % group_size == 0
    ), "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert (
        train_batch_size >= group_size
    ), "train_batch_size must be greater than or equal to group_size"

    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    for idx, batch in enumerate(data_loader):
        print(f"GRPO step {idx}")
        load_policy_into_vllm_instance(policy, vllm)
        policy_old = copy.deepcopy(policy)
        # repeat the batch to match the rollout batch size
        prompts = batch["prompt"]
        ground_truths = batch["ground_truth"]
        repeated_prompts = [elem for elem in prompts for _ in range(group_size)]
        repeated_ground_truths = [
            elem for elem in ground_truths for _ in range(group_size)
        ]
        output_texts = [
            output.outputs[0].text
            for output in vllm.generate(repeated_prompts, sampling_params)
        ]
        print(f"Generated {len(output_texts)} outputs")
        advantages, raw_rewards, _ = run_compute_group_normalized_rewards(
            r1_zero_reward_fn,
            output_texts,
            repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )
        print(f"Calculated {len(advantages)} advantages")

        for train_step in range(n_microbatches_per_rollout_batch):
            micro_batch_prompts = repeated_prompts[
                train_step
                * micro_train_batch_size : (train_step + 1)
                * micro_train_batch_size
            ]
            micro_batch_responses = output_texts[
                train_step
                * micro_train_batch_size : (train_step + 1)
                * micro_train_batch_size
            ]

            tokenization_result = adapters.run_tokenize_prompt_and_output(
                tokenizer=tokenizer,
                prompt_strs=micro_batch_prompts,
                output_strs=micro_batch_responses,
            )

            # Move all tensors to the same device as the model
            tokenization_result = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenization_result.items()
            }

            log_probs = adapters.run_get_response_log_probs(
                model=policy,
                input_ids=tokenization_result["input_ids"],
                labels=tokenization_result["labels"],
                return_token_entropy=False,
            )["log_probs"]

            old_log_probs = adapters.run_get_response_log_probs(
                model=policy_old,
                input_ids=tokenization_result["input_ids"],
                labels=tokenization_result["labels"],
                return_token_entropy=False,
            )["log_probs"]

            loss, _ = run_grpo_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=tokenization_result["response_mask"],
                loss_type=loss_type,
                raw_rewards=raw_rewards,
                advantages=advantages,
                old_log_probs=old_log_probs,
                cliprange=advantage_eps,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )

            if (idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            print(f"Microbatch {train_step} done")

        if eval_frequency != 0 and (idx + 1) % eval_frequency == 0:
            load_policy_into_vllm_instance(policy, vllm)
            evaluate_vllm(
                vllm,
                output_path=f"{EVAL_DIR}/{run_id}/eval_{idx}.jsonl",
            )
        break


if __name__ == "__main__":
    grpo_train_loop(
        rollout_batch_size=8,
        train_batch_size=8,
        gradient_accumulation_steps=1,
    )
