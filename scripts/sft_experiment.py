from unittest.mock import patch

import torch
import wandb

from scripts.evaluate_baseline import evaluate_vllm
from scripts.load_sft_data import get_sft_dataloader
from tests import adapters
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from vllm.entrypoints.llm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed


EVAL_DIR = "/data/users/hadikotaich/cs336-assignment5-alignment/eval_results/sft/"


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


def run_sft(
    training_data_size: int,
    learning_rate: float,
    batch_size: int,
    eval_frequency: int,
    eval_before_train: bool = False,
    gradient_accumulation_steps: int = 1,
):
    run_id = wandb.util.generate_id()
    wandb.init(project="sft", id=run_id)
    print(f"run_id: {run_id}")
    model_id = (
        "/data/users/hadikotaich/cs336-assignment5-alignment/models/Qwen2.5-Math-1.5B"
    )
    vllm = init_vllm(
        model_id=model_id,
        seed=42,
        gpu_memory_utilization=0.85,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
    )

    # not sure if we need this but why not
    load_policy_into_vllm_instance(model, vllm)

    # evaluate before training
    if eval_before_train:
        evaluate_vllm(
            vllm,
            output_path=f"{EVAL_DIR}/{run_id}/before_training.jsonl",
        )

    # load training data
    data_loader = get_sft_dataloader(
        num_rows=training_data_size, batch_size=batch_size, shuffle=False
    )

    for idx, batch in enumerate(data_loader):
        prompts = batch["prompt"]
        responses = batch["response"]

        tokenization_result = adapters.run_tokenize_prompt_and_output(
            tokenizer=tokenizer,
            prompt_strs=prompts,
            output_strs=responses,
        )

        if idx == 0:
            print(f"tokenization_result: {tokenization_result}")

        log_probs = adapters.run_get_response_log_probs(
            model=model,
            input_ids=tokenization_result["input_ids"],
            labels=tokenization_result["labels"],
            return_token_entropy=False,
        )["log_probs"]

        loss, _ = adapters.run_sft_microbatch_train_step(
            policy_log_probs=log_probs,
            response_mask=tokenization_result["response_mask"],
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        print(f"loss norm: {torch.norm(loss)}")

        if (idx + 1) % gradient_accumulation_steps == 0:
            model.optimizer.step()
            model.optimizer.zero_grad()

        if (idx + 1) % eval_frequency == 0:
            load_policy_into_vllm_instance(model, vllm)
            evaluate_vllm(
                vllm,
                output_path=f"{EVAL_DIR}/{run_id}/eval_{idx}.jsonl",
            )


if __name__ == "__main__":
    run_sft(
        training_data_size=128,
        learning_rate=0.0001,
        batch_size=5,
        eval_frequency=8,
        eval_before_train=True,
    )
