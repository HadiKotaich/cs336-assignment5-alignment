from datasets import load_dataset


def load_sft_data(num_rows: int) -> list[dict[str, str]]:
    """Load SFT data from Tulu-3 dataset.

    Args:
        num_rows: Number of rows to load from the dataset

    Returns:
        List of dictionaries with keys 'id', 'prompt', and 'response'
    """
    dataset = load_dataset("allenai/tulu-3-sft-personas-math")
    train_data = dataset["train"]

    result = []
    for i, row in enumerate(train_data):
        if i >= num_rows:
            break

        # Extract prompt and response from messages
        # Tulu format typically has [user_message, assistant_message]
        messages = row["messages"]

        # The prompt is the user's message (first message)
        prompt = messages[0]["content"] if len(messages) > 0 else ""

        # The response is the assistant's message (last message)
        response = messages[-1]["content"] if len(messages) > 1 else ""

        result.append({"id": row["id"], "prompt": prompt, "response": response})

    return result


if __name__ == "__main__":
    # Test the function
    data = load_sft_data(5)
    print(f"Loaded {len(data)} examples")
    print(f"\nFirst example:")
    print(f"ID: {data[0]['id']}")
    print(f"Prompt: {data[0]['prompt']}...")
    print(f"Response: {data[0]['response']}...")
