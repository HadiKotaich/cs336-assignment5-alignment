import re
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


def get_prompt_template() -> str:
    """Load the prompt template used for evaluation."""
    prompt_path = (
        Path(__file__).parent.parent / "cs336_alignment" / "prompts" / "r1_zero.prompt"
    )
    with open(prompt_path) as f:
        return f.read()


def format_response(raw_response: str) -> str:
    """Convert MetaMathQA response to <think> ... </think> <answer> ... </answer> format.

    Args:
        raw_response: Raw response from MetaMathQA, expected to have #### marker

    Returns:
        Formatted response with <think> and <answer> tags
    """
    # Split on #### to separate reasoning from answer
    if "####" not in raw_response:
        raise ValueError("Response must contain #### marker")

    parts = raw_response.split("####")
    reasoning = parts[0].strip()
    answer_full = parts[1].strip()

    # Extract just the numeric answer (first line before "The answer is:")
    answer = answer_full.split("\n")[0].strip()

    # Format as: <think>reasoning</think> <answer>answer</answer>
    return f"<think>\n{reasoning}\n</think> <answer>{answer}</answer>"


class SFTDataset(Dataset):
    """PyTorch Dataset for SFT training data from MetaMathQA."""

    def __init__(self, num_rows: int):
        """Initialize the dataset.

        Args:
            num_rows: Number of rows to load from the dataset
        """
        dataset = load_dataset("meta-math/MetaMathQA")
        train_data = dataset["train"]
        prompt_template = get_prompt_template()

        self.data = []
        count = 0
        for i, row in enumerate(train_data):
            if count >= num_rows:
                break

            # Skip rows where response doesn't have ####
            if "####" not in row["response"]:
                continue

            try:
                # MetaMathQA has 'query' and 'response' fields
                row_id = row.get("id", str(i))
                question = row["query"]

                # Format prompt to match evaluation format
                formatted_prompt = prompt_template.format(question=question)

                # Format response with <think> and <answer> tags
                formatted_response = format_response(row["response"])

                self.data.append(
                    {
                        "id": row_id,
                        "prompt": formatted_prompt,
                        "response": formatted_response,
                    }
                )
                count += 1
            except (ValueError, KeyError) as e:
                # Skip rows that can't be properly formatted
                continue

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.data[idx]


def get_sft_dataloader(
    num_rows: int, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader for SFT training data.

    Args:
        num_rows: Number of rows to load from the dataset
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data

    Returns:
        PyTorch DataLoader
    """
    dataset = SFTDataset(num_rows)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    # Test the dataset and dataloader
    print("Testing SFTDataset...")
    print("\n\nTesting DataLoader with batch_size=3...")
    dataloader = get_sft_dataloader(num_rows=10, batch_size=3, shuffle=False)
    print(f"Number of batches: {len(dataloader)}")

    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  Batch size: {len(batch['id'])}")
        print(f"  IDs: {batch['id']}")
        if i == 0:  # Show all items in first batch
            for j in range(len(batch["id"])):
                print(f"\n  Item {j} in batch:")
                print(f"    ID: {batch['id'][j]}")
                print(f"    Prompt: {batch['prompt'][j]}")
                print(f"    Response: {batch['response'][j]}")
