from automation.pipelines import Pipeline
from automation.tasks import LMEvalTask, LLMCompressorTask


def get_quip_modifier(transform_block_size: int | None):
    from llmcompressor.modifiers.transform import QuIPModifier

    return QuIPModifier(
        transform_type="hadamard", transform_block_size=transform_block_size
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

    # TODO: issue in llm-compressor when loading QuantizationModifiers from generated
    # yaml --> Please specify either `targets` or `config_groups`
    # manually delete for now
    modifier = QuantizationModifier(
        config_groups={"group_0": get_w4a16_scheme(group_size)}, ignore=["lm_head"]
    )
    modifier.targets = None
    return modifier


def get_gptq_modifier(group_size: int = 128):
    from llmcompressor.modifiers.quantization import (
        GPTQModifier,
    )

    modifier = GPTQModifier(
        config_groups={"group_0": get_w4a16_scheme(group_size)}, ignore=["lm_head"]
    )
    modifier.targets = None
    return modifier


recipes = {
    "RTN_W4A16G128": get_rtn_modifier(128),
    "GPTQ_W4A16G128": get_gptq_modifier(128),
    "QUIP_B128_RTN_W4A16G128": [get_quip_modifier(128), get_rtn_modifier(128)],
    "QUIP_B128_GPTQ_W4A16G128": [get_quip_modifier(128), get_gptq_modifier(128)],
    "QUIP_B64_RTN_W4A16G64": [get_quip_modifier(64), get_rtn_modifier(64)],
    "QUIP_B64_GPTQ_W4A16G64": [get_quip_modifier(64), get_gptq_modifier(64)],
}


def average_scores(task):
    gsm8k_score = task.get_reported_scalars()["gsm8k"]["exact_match,strict-match"]["y"][
        0
    ]
    winogrande_score = task.get_reported_scalars()["winogrande"]["acc,none"]["y"][0]
    average_score = (gsm8k_score + winogrande_score) / 2.0
    task.get_logger().report_scalar(
        title="score", series="average", iteration=0, value=average_score
    )


if __name__ == "__main__":
    from llmcompressor.recipe import Recipe

    pipeline = Pipeline(
        project_name="brian_transforms",
        pipeline_name="transforms_benchmark",
        job_end_callback=average_scores,
    )

    for model_id in [
        "meta-llama/Llama-3.2-3B-Instruct",
        # "meta-llama/Llama-3.1-8B-Instruct",
    ]:
        for recipe_id, recipe_modifiers in recipes.items():
            # NOTE: passing recipe in as a list of modifiers results in parsing
            # errors. Use `Recipe.from_modifiers(recipe).model_dump_json()` instead
            recipe = Recipe.from_modifiers(recipe_modifiers)
            compress_step_name = f"compress-{recipe_id}"
            compress_step = LLMCompressorTask(
                project_name="brian_transforms",
                task_name=compress_step_name,
                model_id=model_id,
                text_samples=512,
                recipe=recipe.yaml(),
            )
            compress_step.create_task()

            eval_step = LMEvalTask(
                project_name="brian_transforms",
                task_name=f"eval-{recipe_id}",
                model_id="dummuy",  # overridden
                clearml_model=True,
                tasks=["gsm8k", "winogrande"],
                num_fewshot=5,
                # limit=10,
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
                name=f"eval-{recipe_id}",
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

    pipeline.start()
    # pipeline.execute_locally()
