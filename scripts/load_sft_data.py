import re

from datasets import load_dataset


def clean_response(response: str) -> str:
    """Remove the 'The answer is: X' suffix from MetaMathQA responses.

    Args:
        response: Raw response from the dataset

    Returns:
        Cleaned response without the 'The answer is:' suffix, but keeping '####'
    """
    # Remove only the "The answer is: X" line at the end, keep the "####" part
    response = re.sub(r'\s*The answer is:.*$', '', response, flags=re.MULTILINE)
    return response.strip()


def load_sft_data(num_rows: int) -> list[dict[str, str]]:
    """Load SFT data from MetaMathQA dataset.

    Args:
        num_rows: Number of rows to load from the dataset

    Returns:
        List of dictionaries with keys 'id', 'prompt', and 'response'
    """
    dataset = load_dataset("meta-math/MetaMathQA")
    train_data = dataset["train"]

    result = []
    for i, row in enumerate(train_data):
        if i >= num_rows:
            break

        # MetaMathQA has 'query' and 'response' fields
        # Generate a simple ID if not present in the dataset
        row_id = row.get("id", str(i))

        result.append(
            {
                "id": row_id,
                "prompt": row["query"],
                "response": clean_response(row["response"]),
            }
        )

    return result


if __name__ == "__main__":
    # Test the function
    data = load_sft_data(5)
    print(f"Loaded {len(data)} examples")
    for i in range(5):
        print(f"\nExample {i}:")
        print(f"ID: {data[i]['id']}")
        print(f"Prompt: {data[i]['prompt']}...")
        print(f"Response: {data[i]['response']}...")
