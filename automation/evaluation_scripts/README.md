# Evaluation scripts

Scripts that can be used to launch evaluation runs using ClearML queues.

## lm-evaluation-harness

### Example: evaluating Meta-Llama-3.1-Instruct on Meta's version of ARC-Challenge
```bash
python queue_lm_evaluation_harness_vllm.py \
  --model-id meta-llama/Meta-Llama-3.1-Instruct \
  --queue-name onehsot-a100x1 \
  --project-name "LLM reference" \
  --task-name "Meta-Llama-3.1-Instruct/arc_challenge_llama_3.1_instruct/vllm" \
  --benchmark-tasks arc_challenge_llama_3.1_instruct \
  --num-fewshot 0 \
  --add-bos-token \
  --max-gen-toks 10 \
  --apply-chat-template
```

### Arguments
#### General
- model-id: either HF name, path to /network folder, or ClearML model-id
- queue-name: ClearML queue
- project-name: name of CLearML project
- task-name: name of ClearML task
- clearml-model: if True then model-id is interpreted as ClearML model-id
- benchmark-tasks: lm-evaluation-harness tasks
- num-fewshot: number of few-shot examples
- add-bos-token: if True then bos token is added to prompts
- apply-chat-template: if True then chat template is applied to prompts
- fewshot-as-multiturn: if True then few-shot examples are treated as multi-turn chat
- batch-size: batch size
- trust-remote-code: if True then allow custom model definition
- packages: pypi packages to be installed in addition to standard ones

#### vLLM version
- max-gen-toks: maximum number of generated tokens (for vllm version)
- gpu-memory-utilization: memory to be pre-allocated for kv-cache
- max-model-len: maximum sequence length
- build-vllm: if True then vLLM is installed from source

#### SparseML / HF versions
- max-length: maximum sequence length
- parallelize: if True, then shard model across available GPUs (using accelerate)

## evalplus

### Example: evaluating Meta-Llama-3.1-Instruct on HumanEval
```bash
python queue_evalplus_vllm.py \
  --model-id meta-llama/Meta-Llama-3.1-Instruct \
  --queue-name onehsot-a100x1 \
  --project-name "LLM reference" \
  --task-name "Meta-Llama-3.1-Instruct/humaneval/vllm" \
  --benchmark-task humaneval
```

### Arguments
#### General
- model-id: either HF name, path to /network folder, or ClearML model-id
- queue-name: ClearML queue
- project-name: name of CLearML project
- task-name: name of ClearML task
- clearml-model: if True then model-id is interpreted as ClearML model-id
- benchmark-task: evalplus task
- batch-size: batch size
- disable-sanitize: if True then generated code supplements are not sanitized before evaluation 
- num-samples: number of code supplements to be generated per problem
- temperature: generation temperature
- trust-remote-code: if True then allow custom model definition
- packages: pypi packages to be installed in addition to standard ones

#### vLLM version
- build-vllm: if True then vLLM is installed from source

### Note
- HF and SparseML versions only support single-GPU evaluations at the moment