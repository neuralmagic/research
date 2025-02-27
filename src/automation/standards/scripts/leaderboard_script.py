from automation.tasks.scripts.lmeval_script import main as lmeval_main
from clearml import Task

def compute_leaderboard_ifeval(results):
    inst_level_strict_acc = results["results"]["leaderboard_ifeval"]["inst_level_strict_acc,none"]
    prompt_level_strict_acc = results["results"]["leaderboard_ifeval"]["prompt_level_strict_acc,none"]
    average_score = (inst_level_strict_acc + prompt_level_strict_acc) / 2 
    return average_score

def compute_leaderboard_math_hard(results):
    subtasks = [
        "leaderboard_math_algebra_hard",
        "leaderboard_math_counting_and_prob_hard",
        "leaderboard_math_geometry_hard",
        "leaderboard_math_intermediate_algebra_hard",
        "leaderboard_math_num_theory_hard",
        "leaderboard_math_prealgebra_hard",
        "leaderboard_math_precalculus_hard"
    ]
    exact_match_scores = []
    for subtask in subtasks:
        score = results["results"][subtask]["exact_match,none"]
        exact_match_scores.append(score)
    average_score = sum(exact_match_scores) / len(exact_match_scores)
    return average_score

def compute_leaderboard_musr(results):
    subtasks = {
        "leaderboard_musr_murder_mysteries": 2,
        "leaderboard_musr_object_placements": 5,
        "leaderboard_musr_team_allocation": 3
    }
    adjusted_accuracies = []
    for subtask, num_choices in subtasks.items():
        actual_acc = results["results"][subtask]["acc_norm,none"]
        random_baseline = 1 / num_choices
        adjusted_acc = max(actual_acc - random_baseline, 0) / (1 - random_baseline)
        adjusted_accuracies.append(adjusted_acc)
    average_score = sum(adjusted_accuracies) / len(adjusted_accuracies)
    return average_score

def raw_compute_leaderboard_musr(results):
    subtasks = {
        "leaderboard_musr_murder_mysteries": 2,
        "leaderboard_musr_object_placements": 5,
        "leaderboard_musr_team_allocation": 3
    }
    raw_accuracies = []
    for subtask, _ in subtasks.items():
        raw_accuracies.append(results["results"][subtask]["acc_norm,none"])
    average_score = sum(raw_accuracies) / len(raw_accuracies)
    return average_score

def compute_leaderboard_bbh(results):
    subtasks_choices = {
        "leaderboard_bbh_sports_understanding": 2,
        "leaderboard_bbh_tracking_shuffled_objects_three_objects": 3,
        "leaderboard_bbh_navigate": 2,
        "leaderboard_bbh_snarks": 2,
        "leaderboard_bbh_date_understanding": 6,
        "leaderboard_bbh_reasoning_about_colored_objects": 18,
        "leaderboard_bbh_object_counting": 19,
        "leaderboard_bbh_logical_deduction_seven_objects": 7,
        "leaderboard_bbh_geometric_shapes": 11,
        "leaderboard_bbh_web_of_lies": 2,
        "leaderboard_bbh_movie_recommendation": 6,
        "leaderboard_bbh_logical_deduction_five_objects": 5,
        "leaderboard_bbh_salient_translation_error_detection": 6,
        "leaderboard_bbh_disambiguation_qa": 3,
        "leaderboard_bbh_temporal_sequences": 4,
        "leaderboard_bbh_hyperbaton": 2,
        "leaderboard_bbh_logical_deduction_three_objects": 3,
        "leaderboard_bbh_causal_judgement": 2,
        "leaderboard_bbh_formal_fallacies": 2,
        "leaderboard_bbh_tracking_shuffled_objects_seven_objects": 7,
        "leaderboard_bbh_ruin_names": 6,
        "leaderboard_bbh_penguins_in_a_table": 5,
        "leaderboard_bbh_boolean_expressions": 2,
        "leaderboard_bbh_tracking_shuffled_objects_five_objects": 5
    }
    adjusted_accuracies = []
    for subtask, num_choices in subtasks_choices.items():
        actual_acc = results["results"][subtask]["acc_norm,none"]
        random_baseline = 1 / num_choices
        adjusted_acc = max(actual_acc - random_baseline, 0) / (1 - random_baseline)
        adjusted_accuracies.append(adjusted_acc)
    average_score = sum(adjusted_accuracies) / len(adjusted_accuracies)
    return average_score

def raw_compute_leaderboard_bbh(results):
    subtasks_choices = {
        "leaderboard_bbh_sports_understanding": 2,
        "leaderboard_bbh_tracking_shuffled_objects_three_objects": 3,
        "leaderboard_bbh_navigate": 2,
        "leaderboard_bbh_snarks": 2,
        "leaderboard_bbh_date_understanding": 6,
        "leaderboard_bbh_reasoning_about_colored_objects": 18,
        "leaderboard_bbh_object_counting": 19,
        "leaderboard_bbh_logical_deduction_seven_objects": 7,
        "leaderboard_bbh_geometric_shapes": 11,
        "leaderboard_bbh_web_of_lies": 2,
        "leaderboard_bbh_movie_recommendation": 6,
        "leaderboard_bbh_logical_deduction_five_objects": 5,
        "leaderboard_bbh_salient_translation_error_detection": 6,
        "leaderboard_bbh_disambiguation_qa": 3,
        "leaderboard_bbh_temporal_sequences": 4,
        "leaderboard_bbh_hyperbaton": 2,
        "leaderboard_bbh_logical_deduction_three_objects": 3,
        "leaderboard_bbh_causal_judgement": 2,
        "leaderboard_bbh_formal_fallacies": 2,
        "leaderboard_bbh_tracking_shuffled_objects_seven_objects": 7,
        "leaderboard_bbh_ruin_names": 6,
        "leaderboard_bbh_penguins_in_a_table": 5,
        "leaderboard_bbh_boolean_expressions": 2,
        "leaderboard_bbh_tracking_shuffled_objects_five_objects": 5
    }
    raw_accuracies = []
    for subtask, _ in subtasks_choices.items():
        raw_accuracies.append(results["results"][subtask]["acc_norm,none"])
    average_score = sum(raw_accuracies) / len(raw_accuracies)
    return average_score

def compute_leaderboard_gpqa(results):
    subtasks = [
        "leaderboard_gpqa_main",
        "leaderboard_gpqa_diamond",
        "leaderboard_gpqa_extended"
    ]
    adjusted_accuracies = []
    num_choices = 4
    for subtask in subtasks:
        actual_acc = results["results"][subtask]["acc_norm,none"]
        random_baseline = 1 / num_choices
        adjusted_acc = max(actual_acc - random_baseline, 0) / (1 - random_baseline)
        adjusted_accuracies.append(adjusted_acc)
    average_score = sum(adjusted_accuracies) / len(adjusted_accuracies)
    return average_score

def raw_compute_leaderboard_gpqa(results):
    subtasks = [
        "leaderboard_gpqa_main",
        "leaderboard_gpqa_diamond",
        "leaderboard_gpqa_extended"
    ]
    raw_accuracies = []
    for subtask in subtasks:
        raw_accuracies.append(results["results"][subtask]["acc_norm,none"])
    average_score = sum(raw_accuracies) / len(raw_accuracies)
    return average_score

def compute_leaderboard_mmlu_pro(results):
    num_choices = 10
    random_baseline = 1 / num_choices
    adjusted_acc = max(results["results"]["leaderboard_mmlu_pro"]["acc,none"] - random_baseline, 0) / (1 - random_baseline)
    return adjusted_acc

def raw_compute_leaderboard_mmlu_pro(results):
    return results["results"]["leaderboard_mmlu_pro"]["acc,none"]

def main():
    results = lmeval_main()

    ifeval = compute_leaderboard_ifeval(results)
    math = compute_leaderboard_math_hard(results)
    musr = compute_leaderboard_musr(results)
    bbh = compute_leaderboard_bbh(results)
    gpqa = compute_leaderboard_gpqa(results)
    mmlu_pro = compute_leaderboard_mmlu_pro(results)
    leaderboard = (ifeval + math + musr + bbh + gpqa + mmlu_pro) / 6.0

    print("Final Scores:")
    print(f"Leaderboard_ifeval: {ifeval:.2f}%")
    print(f"Leaderboard_bbh: {bbh:.2f}%")
    print(f"Leaderboard_math_hard: {math:.2f}%")
    print(f"Leaderboard_gpqa: {gpqa:.2f}%")
    print(f"Leaderboard_musr: {musr:.2f}%")
    print(f"Leaderboard_mmlu_pro: {mmlu_pro:.2f}%")
    print(f"Average: {leaderboard:.2f}%")

    print(f"\nFor copy-paste into Google Sheet: \n{ifeval:.2f}|{bbh:.2f}|{math:.2f}|{gpqa:.2f}|{musr:.2f}|{mmlu_pro:.2f}")

    task = Task.current_task()
    task.get_logger().report_single_value(name="ifeval", value=ifeval)
    task.get_logger().report_single_value(name="math_hard", value=math)
    task.get_logger().report_single_value(name="musr", value=musr)
    task.get_logger().report_single_value(name="bbh", value=bbh)
    task.get_logger().report_single_value(name="gpqa", value=gpqa)
    task.get_logger().report_single_value(name="mmlu_pro", value=mmlu_pro)
    task.get_logger().report_single_value(name="leaderboard", value=leaderboard)

    task.get_logger().report_scalar(title="leaderboard", series="average", iteration=0, value=leaderboard)
    task.get_logger().report_scalar(title="ifeval", series="normalized", iteration=0, value=ifeval)
    task.get_logger().report_scalar(title="math_hard", series="normalized", iteration=0, value=math)
    task.get_logger().report_scalar(title="musr", series="normalized", iteration=0, value=musr)
    task.get_logger().report_scalar(title="bbh", series="normalized", iteration=0, value=bbh)
    task.get_logger().report_scalar(title="gpqa", series="normalized", iteration=0, value=gpqa)
    task.get_logger().report_scalar(title="mmlu_pro", series="normalized", iteration=0, value=mmlu_pro)

    if len(task.get_models()["input"]) == 1:
        clearml_model_handle = task.get_models()["input"][0]
        clearml_model_handle.report_single_value(name="ifeval", value=ifeval)
        clearml_model_handle.report_single_value(name="math_hard", value=math)
        clearml_model_handle.report_single_value(name="musr", value=musr)
        clearml_model_handle.report_single_value(name="bbh", value=bbh)
        clearml_model_handle.report_single_value(name="gpqa", value=gpqa)
        clearml_model_handle.report_single_value(name="mmlu_pro", value=mmlu_pro)
        clearml_model_handle.report_single_value(name="leaderboard", value=leaderboard)

if __name__ == "__main__":
    main()