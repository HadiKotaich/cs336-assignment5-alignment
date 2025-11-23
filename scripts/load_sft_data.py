import re

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


def clean_response(response: str) -> str:
    """Remove the 'The answer is: X' suffix from MetaMathQA responses.

    Args:
        response: Raw response from the dataset

    Returns:
        Cleaned response without the 'The answer is:' suffix, but keeping '####'
    """
    # Remove only the "The answer is: X" line at the end, keep the "####" part
    response = re.sub(r"\s*The answer is:.*$", "", response, flags=re.MULTILINE)
    return response.strip()


class SFTDataset(Dataset):
    """PyTorch Dataset for SFT training data from MetaMathQA."""

    def __init__(self, num_rows: int):
        """Initialize the dataset.

        Args:
            num_rows: Number of rows to load from the dataset
        """
        dataset = load_dataset("meta-math/MetaMathQA")
        train_data = dataset["train"]

        self.data = []
        count = 0
        for i, row in enumerate(train_data):
            if count >= num_rows:
                break

            # MetaMathQA has 'query' and 'response' fields
            row_id = row.get("id", str(i))
            cleaned_response = clean_response(row["response"])

            # Skip rows where response doesn't have ####
            if "####" not in cleaned_response:
                continue

            self.data.append(
                {
                    "id": row_id,
                    "prompt": row["query"],
                    "response": cleaned_response,
                }
            )
            count += 1

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
