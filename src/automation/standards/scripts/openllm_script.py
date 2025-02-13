from automation.tasks.scripts.lmeval_script import main as lmeval_main
from clearml import Task

def main():
    lmeval_main()
    task = Task.current_task()

    arc_challenge_score = task.get_reported_single_value("arc_challenge/25shot/acc,none")
    gsm8k_score = task.get_reported_single_value("gsm8k/5shot/exact_match,strict-match")
    hellaswag_score = task.get_reported_single_value("hellaswag/10shot/acc_norm,none")
    mmlu_score = task.get_reported_single_value("mmlu/acc,none")
    winogrande_score = task.get_reported_single_value("winogrande/5shot/acc,none")
    truthfulqa_score = task.get_reported_single_value("truthfulqa_mc2/0shot/acc,none")

    openllm_score = (arc_challenge_score + gsm8k_score + hellaswag_score + mmlu_score + winogrande_score + truthfulqa_score) / 6

    task.get_logger().report_single_value(name="openllm", value=openllm_score)

if __name__ == '__main__':
    main()