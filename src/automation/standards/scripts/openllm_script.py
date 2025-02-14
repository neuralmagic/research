from automation.tasks.scripts.lmeval_script import main as lmeval_main
from clearml import Task

def main():
    results = lmeval_main()

    arc_challenge_score = results["results"]["arc_challenge"]["acc,none"]
    gsm8k_score = results["results"]["gsm8k"]["exact_match,strict-match"]
    hellaswag_score = results["results"]["hellaswag"]["acc_norm,none"]
    mmlu_score = results["results"]["mmlu"]["acc,none"]
    winogrande_score = results["results"]["winogrande"]["acc,none"]
    truthfulqa_score = results["results"]["truthfulqa_mc2"]["acc,none"]

    openllm_score = (arc_challenge_score + gsm8k_score + hellaswag_score + mmlu_score + winogrande_score + truthfulqa_score) / 6.

    task = Task.current_task()
    task.get_logger().report_single_value(name="openllm", value=openllm_score)

if __name__ == '__main__':
    main()