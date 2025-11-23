from unittest.mock import patch

import torch

import wandb
from load_sft_data import get_sft_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from vllm.entrypoints.llm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed


# Setup wandb metrics
wandb.init()
wandb.define_metric("train_step")  # the x‑axis for training
wandb.define_metric("eval_step")  # the x‑axis for evaluation
# everything that starts with train/ is tied to train_step
wandb.define_metric("train/*", step_metric="train_step")
# everything that starts with eval/ is tied to eval_step
wandb.define_metric("eval/*", step_metric="eval_step")


def init_vllm(model_id: str, seed: int, gpu_memory_utilization: float = 0.85) -> LLM:
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.

    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def run_sft(training_data_size: int, learning_rate: float, batch_size: int):
    model_id = (
        "/data/users/hadikotaich/cs336-assignment5-alignment/models/Qwen2.5-Math-1.5B"
    )
    vllm = init_vllm(
        model_id=model_id,
        seed=42,
        gpu_memory_utilization=0.85,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
    )

    # not sure if we need this but why not
    load_policy_into_vllm_instance(model, vllm)

    # load training data
    data_loader = get_sft_dataloader(
        num_rows=training_data_size, batch_size=batch_size, shuffle=False
    )

    for idx, batch in enumerate(data_loader):
        prompts = batch['prompt']
        responses = batch['response']
        print(f"Batch {idx} of {len(data_loader)}")
        print(f"len(prompts) = {len(prompts)}, len(responses) = {len(responses)}")
        if idx == 0:
            for i in range(len(prompts)):
                print(f"prompt {i}: {prompts[i]}")
                print(f"response {i}: {responses[i]}")


if __name__ == "__main__":
    run_sft(training_data_size=128, learning_rate=0.0001, batch_size=5)
