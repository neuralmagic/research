from typing import Literal
from clearml import Task

from automation.pipelines import Pipeline
from automation.tasks import LMEvalTask, LLMCompressorTask

PROJECT_NAME = "brian_transforms_v1"


def get_spinquant_modifier(
    transform_block_size: int | None,
    rotations: list[Literal["R1", "R2", "R4"]] = ["R1", "R2"],
):
    from llmcompressor.modifiers.transform import SpinQuantModifier

    return SpinQuantModifier(
        transform_type="hadamard",
        transform_block_size=transform_block_size,
        rotations=rotations,
    )


def get_quip_modifier(
    transform_block_size: int | None, rotations: list[Literal["u", "v"]] = ["u", "v"]
):
    from llmcompressor.modifiers.transform import QuIPModifier

    return QuIPModifier(
        transform_type="hadamard",
        transform_block_size=transform_block_size,
        rotations=rotations,
    )


def get_w4a16_scheme(group_size: int = 128):
    from compressed_tensors.quantization import (
        QuantizationScheme,
        QuantizationStrategy,
        QuantizationType,
        QuantizationArgs,
    )

    return QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.GROUP,
            group_size=group_size,
            symmetric=True,
            dynamic=False,
        ),
    )


def get_rtn_modifier(group_size: int = 128):
    from llmcompressor.modifiers.quantization import (
        QuantizationModifier,
    )

    return QuantizationModifier(
        config_groups={"group_0": get_w4a16_scheme(group_size)}, ignore=["lm_head"]
    )


def get_gptq_modifier(group_size: int = 128):
    from llmcompressor.modifiers.quantization import (
        GPTQModifier,
    )

    return GPTQModifier(
        config_groups={"group_0": get_w4a16_scheme(group_size)}, ignore=["lm_head"]
    )


recipes = {
    "DENSE": [],
    # "RTN_W4A16G128": get_rtn_modifier(128),
    # "GPTQ_W4A16G128": get_gptq_modifier(128),
    # "QUIPv_B128_RTN_W4A16G128": [get_quip_modifier(128, ["v"]), get_rtn_modifier(128)],
    # "QUIPv_B128_GPTQ_W4A16G128": [
    #     get_quip_modifier(128, ["v"]),
    #     get_gptq_modifier(128),
    # ],
    # "QUIPuv_B128_RTN_W4A16G128": [
    #     get_quip_modifier(128, ["u", "v"]),
    #     get_rtn_modifier(128),
    # ],
    # "QUIPuv_B128_GPTQ_W4A16G128": [
    #     get_quip_modifier(128, ["u", "v"]),
    #     get_gptq_modifier(128),
    # ],
    "SpinQuantR1R2_B128_GPTQ_W4A16G128": [
        get_spinquant_modifier(128, ["R1", "R2"]),
        get_gptq_modifier(128),
    ],
    "SpinQuantR1R2R4_B128_GPTQ_W4A16G128": [
        get_spinquant_modifier(128, ["R1", "R2", "R4"]),
        get_gptq_modifier(128),
    ],
    # "RTN_W4A16G64": get_rtn_modifier(64),
    # "GPTQ_W4A16G64": get_gptq_modifier(64),
    # "QUIPv_B64_RTN_W4A16G64": [get_quip_modifier(64, ["v"]), get_rtn_modifier(64)],
    # "QUIPv_B64_GPTQ_W4A16G64": [
    #     get_quip_modifier(64, ["v"]),
    #     get_gptq_modifier(64),
    # ],
    # "QUIPuv_B64_RTN_W4A16G64": [get_quip_modifier(64, ["u", "v"]), get_rtn_modifier(64)],
    # "QUIPuv_B64_GPTQ_W4A16G64": [
    #     get_quip_modifier(64, ["u", "v"]),
    #     get_gptq_modifier(64),
    # ],
    "SpinQuantR1R2_B64_GPTQ_W4A16G64": [
        get_spinquant_modifier(64, ["R1", "R2"]),
        get_gptq_modifier(64),
    ],
    "SpinQuantR1R2R4_B64_GPTQ_W4A16G64": [
        get_spinquant_modifier(64, ["R1", "R2", "R4"]),
        get_gptq_modifier(64),
    ],
    # "RTN_W4A16G32": get_rtn_modifier(32),
    # "GPTQ_W4A16G32": get_gptq_modifier(32),
    # "QUIPv_B32_RTN_W4A16G32": [get_quip_modifier(32, ["v"]), get_rtn_modifier(32)],
    # "QUIPv_B32_GPTQ_W4A16G32": [
    #     get_quip_modifier(32, ["v"]),
    #     get_gptq_modifier(32),
    # ],
    # "QUIPuv_B32_RTN_W4A16G32": [get_quip_modifier(32, ["u", "v"]), get_rtn_modifier(32)],
    # "QUIPuv_B32_GPTQ_W4A16G32": [
    #     get_quip_modifier(32, ["u", "v"]),
    #     get_gptq_modifier(32),
    # ],
    "SpinQuantR1R2_B32_GPTQ_W4A16G32": [
        get_spinquant_modifier(32, ["R1", "R2"]),
        get_gptq_modifier(32),
    ],
    "SpinQuantR1R2R4_B32_GPTQ_W4A16G32": [
        get_spinquant_modifier(32, ["R1", "R2", "R4"]),
        get_gptq_modifier(32),
    ],
}


if __name__ == "__main__":
    from llmcompressor.recipe import Recipe

    pipeline = Pipeline(
        project_name=PROJECT_NAME,
        pipeline_name=f"{PROJECT_NAME}_pipeline",
    )

    for model_id in [
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]:
        model_name = model_id.split("/")[-1].replace(".", "_").replace("-", "_")
        for recipe_id, recipe_modifiers in recipes.items():
            # NOTE: passing recipe in as a list of modifiers results in parsing
            # errors. Use `Recipe.from_modifiers(recipe).model_dump_json()` instead
            recipe = Recipe.from_modifiers(recipe_modifiers)
            compress_step_name = f"compress--{model_name}--{recipe_id}"
            compress_step = LLMCompressorTask(
                project_name=PROJECT_NAME,
                task_name=compress_step_name,
                model_id=model_id,
                text_samples=512,
                recipe=recipe.yaml(),
            )
            compress_step.create_task()

            # NOTE: lm_eval settings set to match those found in
            # src/automation/standards/evaluations/openllm.yaml
            # apply_chat_template set to False
            # anmarques: "We notice that apply_chat_template tends to mess up
            # loglikelihood-based evals, which are most of the openllm benchmarks
            # (the model tends to blab before predicting the answer)""
            eval_step = LMEvalTask(
                project_name=PROJECT_NAME,
                task_name=f"eval--{model_name}--{recipe_id}",
                model_id="dummuy",  # overridden
                clearml_model=True,
                tasks=[
                    # openllm tasks + llama variants
                    "arc_challenge",
                    "gsm8k",
                    "hellaswag",
                    "mmlu",
                    "winogrande",
                    "truthfulqa_mc2",
                    "arc_challenge_llama",
                    "gsm8k_llama",
                    # TODO: PPL based metrics broken in lm_eval+vllm
                    # https://github.com/EleutherAI/lm-evaluation-harness/issues/3134
                    # "wikitext"
                ],
                num_fewshot=5,
                apply_chat_template=False,
                model_args=(
                    "gpu_memory_utilization=0.4,dtype=auto,max_model_len=4096,"
                    "add_bos_token=True,enable_chunked_prefill=True"
                ),
                batch_size="auto",
            )
            eval_step.create_task()

            pipeline.add_step(
                name=compress_step_name,
                base_task_id=compress_step.id,
                execution_queue="oneshot-a100x1",
                monitor_models=[
                    compress_step.get_arguments()["Args"]["save_directory"]
                ],
                monitor_artifacts=["recipe"],
            )

            pipeline.add_step(
                name=f"eval-{model_name}-{recipe_id}",
                base_task_id=eval_step.id,
                parents=[compress_step_name],
                execution_queue="oneshot-a100x1",
                parameter_override={
                    "Args/model_id": "${" + compress_step_name + ".models.output.-1.id}"
                },
                monitor_metrics=[
                    ("gsm8k", "exact_match,strict-match"),
                    ("winogrande", "acc,none"),
                ],
            )

    pipeline.execute_remotely()
