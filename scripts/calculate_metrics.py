import json

from evaluate_baseline import EvalRow

#  python ./scripts/calculate_metrics.py

# nb correct with both format and answer reward 1
# nb format correct and answer reward 0
# nb format 0 and answer 0

correct_with_format_reward_1_and_answer_reward_1 = 0
correct_with_format_reward_1_and_answer_reward_0 = 0
correct_with_format_reward_0_and_answer_reward_0 = 0

with open("eval_results/baseline_metrics.jsonl", "r") as f:
    lines = f.readlines()
    for line in lines:
        eval_row = EvalRow.model_validate(json.loads(line))
        if (
            eval_row.metrics["format_reward"] == 1
            and eval_row.metrics["answer_reward"] == 1
        ):
            correct_with_format_reward_1_and_answer_reward_1 += 1
        elif (
            eval_row.metrics["format_reward"] == 1
            and eval_row.metrics["answer_reward"] == 0
        ):
            correct_with_format_reward_1_and_answer_reward_0 += 1
        elif (
            eval_row.metrics["format_reward"] == 0
            and eval_row.metrics["answer_reward"] == 0
        ):
            correct_with_format_reward_0_and_answer_reward_0 += 1


print(
    f"correct_with_format_reward_1_and_answer_reward_1: {correct_with_format_reward_1_and_answer_reward_1}"
)

print(
    f"correct_with_format_reward_1_and_answer_reward_0: {correct_with_format_reward_1_and_answer_reward_0}"
)

print(
    f"correct_with_format_reward_0_and_answer_reward_0: {correct_with_format_reward_0_and_answer_reward_0}"
)
