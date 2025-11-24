import os
from unittest.mock import patch

import torch
import wandb
from scripts.evaluate_baseline import evaluate_vllm
from scripts.load_sft_data import get_sft_dataloader
from tests import adapters
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from vllm.entrypoints.llm import LLM
from vllm.model_executor import set_random_seed as vllm_set_random_seed


EVAL_DIR = "/data/users/hadikotaich/cs336-assignment5-alignment/eval_results/sft/"


def init_vllm(
    model_id: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
) -> LLM:
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
            tensor_parallel_size=tensor_parallel_size,
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
    model: PreTrainedModel,
    data_loader: DataLoader,
    model_id: str,
    run_id: str,
    learning_rate: float,
    eval_frequency: int,  # zero for no eval
    eval_before_train: bool = False,
    gradient_accumulation_steps: int = 1,
):
    wandb.init(project="sft", id=run_id)
    print(f"run_id: {run_id}")
    if eval_before_train or eval_frequency > 0:
        # Use cuda:0 for vLLM (physical GPU 1 when CUDA_VISIBLE_DEVICES=1,2)
        vllm = init_vllm(
            model_id=model_id,
            seed=42,
            gpu_memory_utilization=0.75,
            tensor_parallel_size=1,
        )
        evaluate_vllm(
            vllm,
            output_path=f"{EVAL_DIR}/{run_id}/before_training.jsonl",
        )
    else:
        vllm = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )
    optimizer.zero_grad()

    # not sure if we need this but why not
    load_policy_into_vllm_instance(model, vllm)

    # Clear any residual gradients before training
    optimizer.zero_grad()

    for idx, batch in enumerate(data_loader):
        prompts = batch["prompt"]
        responses = batch["response"]

        tokenization_result = adapters.run_tokenize_prompt_and_output(
            tokenizer=tokenizer,
            prompt_strs=prompts,
            output_strs=responses,
        )

        # Move all tensors to the same device as the model
        tokenization_result = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in tokenization_result.items()
        }

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if vllm is not None and (idx + 1) % eval_frequency == 0:
            load_policy_into_vllm_instance(model, vllm)
            evaluate_vllm(
                vllm,
                output_path=f"{EVAL_DIR}/{run_id}/eval_{idx}.jsonl",
            )

    model_name = os.path.basename(model_id)
    save_dir = os.path.join(os.path.dirname(model_id), f"{model_name}-sft-{run_id}")
    print(f"Saving final model to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved successfully to {save_dir}")


if __name__ == "__main__":
    # Use cuda:1 for training model (physical GPU 2 when CUDA_VISIBLE_DEVICES=1,2)
    model_id = (
        "/data/users/hadikotaich/cs336-assignment5-alignment/models/Qwen2.5-Math-1.5B"
    )
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    training_data_size = 16384
    batch_size = 2  # Microbatch size (actual samples processed at once)
    learning_rate = 0.0001
    run_id = (
        f"sft_sz{training_data_size}_lr_{learning_rate}_bs_{batch_size}_"
        + wandb.util.generate_id()
    )
    data_loader = get_sft_dataloader(
        num_rows=training_data_size, batch_size=batch_size, shuffle=False
    )
    run_sft(
        model=model,
        data_loader=data_loader,
        model_id=model_id,
        run_id=run_id,
        learning_rate=learning_rate,
        eval_frequency=256,
        eval_before_train=True,
        gradient_accumulation_steps=8,  # Accumulate 8 microbatches → effective batch size = 2 * 8 = 16
    )
