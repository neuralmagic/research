from automation.tasks import LightEvalTask

"""
Allowed model_args:
  revision: str = "main"  # revision of the model
  dtype: str = "bfloat16"
  tensor_parallel_size: PositiveInt = 1  # how many GPUs to use for tensor parallelism
  data_parallel_size: PositiveInt = 1  # how many GPUs to use for data parallelism
  pipeline_parallel_size: PositiveInt = 1  # how many GPUs to use for pipeline parallelism
  gpu_memory_utilization: NonNegativeFloat = 0.9  # lower this if you are running out of memory
  max_model_length: PositiveInt | None = None  # maximum length of the model, ussually infered automatically. reduce this if you encouter OOM issues, 4096 is usually enough
  swap_space: PositiveInt = 4  # CPU swap space size (GiB) per GPU.
  seed: NonNegativeInt = 1234
  trust_remote_code: bool = False
  use_chat_template: bool = False
  add_special_tokens: bool = True
  multichoice_continuations_start_space: bool = (
      True  # whether to add a space at the start of each continuation in multichoice generation
  )
  pairwise_tokenization: bool = False  # whether to tokenize the context and continuation separately or together.
  max_num_seqs: PositiveInt = 128  # maximum number of sequences per iteration; This variable and `max_num_batched_tokens` effectively control the batch size at prefill stage. See https://github.com/vllm-project/vllm/issues/2492 for detailed explaination.
  max_num_batched_tokens: PositiveInt = 2048  # maximum number of tokens per batch
  subfolder: str | None = None
"""

model_args="""
model_parameters:
  max_model_length: 40960
  generation_parameters:
    max_new_tokens: 32768
    temperature: 0.6
    top_k: 20
    min_p: 0.0
    top_p: 0.95
"""

task = LightEvalTask(
    project_name="alexandre_debug",
    task_name="test_aime2024_task",
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    config="aime2024",
    model_args=model_args,
)

task.execute_remotely("oneshot-a100x1")
#task.execute_locally()