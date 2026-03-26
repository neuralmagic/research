# General instructions

# Evaluations using vLLM server

# Start vLLM server

Start vLLM server using the configurations recommended by the model provider:

```
vllm serve <model_name> ...
```

Please note arguments such as

- load-format  
- config-format  
- tokenizer-mode  
- tool-call-parser  
- reasoning-parser

# lighteval

**Note**: we’ve identified a couple of bugs in the upstream lighteval library, and until our bug fixes are merged into main (we’ll update this doc to indicate that), please use our own fork of lighteval from the [\`eldar-fix-litellm\` branch](https://github.com/neuralmagic/lighteval/tree/eldar-fix-litellm)

To use lighteval with a vLLM server, one needs to use the litellm endpoint as follows.

## Create yaml file containing configuration for evaluation

**litellm\_config.yaml**

```
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/<model_name>"
  base_url: "http://0.0.0.0:8000/v1"
  api_key: ""
  timeout: 1200
  max_model_lentgh: 96000
  concurrent_requests: 16
  generation_parameters:
    temperature: 0.6
    max_new_tokens: 65536
    top_p: 0.95
    seed: 0
    top_k: 20
    presence_penalty: 1.5
```

**Notes:**

* The model\_name has to be preceded by “hosted\_vllm”. Examples:  
  * HF model: moonshotai/Kimi-K2-Thinking  
    * model\_name: “hosted\_vllm/moonshotai/Kimi-K2-Thinking”  
  * Local model: /local-dir  
    * model\_name: “hosted\_vllm/local-dir”  
* Generation parameters (temperature, top\_p, top\_k, etc) should match the parameters suggested by the model provider, normally listed in the model card and/or *generation\_config.json*.  
* max\_model\_length needs to be specified if the model’s default length is overridden when launching the vLLM server.  
* lighteval allows generating multiple responses per sample, each using a different seed. This is useful when repeating the same evaluation is needed (see syntax below). If using this, each request will spin multiple decoding processes within vLLM. Be mindful of that when determining concurrent\_requests.  
* The seed is specified in the generation\_parameters.  
* The timeout parameter controls the time in seconds per request. This is optional, but I recommend setting it to a large number for reasoning evals.  
* Default value for concurrent\_requests is low (10), so feel free to increase concurrency for better speed.

## Evaluation command

```shell
lighteval endpoint litellm litellm_config.yaml \
"aime25@<k>@<n>|0,math_500@<k>@<n>|0" \
--output-dir <output-dir> \
--save-details
```

   
Notes:

* Tasks are specified as: \<task\_name\>@\<k\>@\<n\>|\<num\_fewshot\>. In previous versions there was a need to specify an additional \<suite\> entry (\<suite\>|\<task\_name\>|\<num\_fewshot\>) that is now deprecated. Examples  
  * aime25@1@8|0  
  * math\_500@1@3|0  
  * gpqa:diamond@1@3|0  
* k and n refer to pass@k with n samples.  
  * This can be used to average results over multiple random seeds. For instance, setting k=1 and n=3 will average the results over 3 different random seeds.  
  * If k and n are not specified, they default to k=1 and n=1.  
* If you are running evals in a loop and need to change some values in *litellm\_config.yaml*, instead of creating a new yaml file for each new configuration, you can also use an inline string instead of the yaml file as an input. For example: 

```shell
lighteval endpoint litellm \
"model_name=hosted_vllm/${SERVED_MODEL_NAME},provider=hosted_vllm,base_url=http://0.0.0.0:${PORT}/v1,timeout=3600,concurrent_requests=8,generation_parameters={temperature:${TEMP},max_new_tokens:${MAX_NEW_TOKENS},top_p:${TOP_P},seed:${SEED},presence_penalty:${PRESENCE_PENALTY},top_k:${TOP_K},min_p:${MIN_P},repetition_penalty:${REPETITION_PENALTY}}" \
"aime25@k=${K}@n=${N_AIME}|0,math_500@k=${K}@n=${N_OTHERS}|0,gpqa:diamond@k=${K}@n=${N_OTHERS}|0,lcb:codegeneration_v6|0" \
--output-dir ${OUTPUT_DIR} \
--save-details
```

# lm-eval generative tasks

## Evaluation command

```shell
lm_eval --model local-chat-completions \
  --tasks gsm8k \
  --model_args "model=<model_name>,max_length=<max_length>,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 5 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path results_gsm8k.json \
  --seed 1234 \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=1.0,top_k=20,max_gen_toks=64000,seed=1234"
```

**Notes:**

* MUST set max\_length explicitly (default=2048 is very low).  
* Greedy decoding is the default. MUST set gen\_kwargs if one wishes to use something other than greedy (and must set do\_sample to True explicitly).  
* Generation parameters (temperature, top\_p, top\_k, etc) should match the parameters suggested by the model provider, normally listed in the model card and/or *generation\_config.json*.  
* Seed MUST be included in the gen\_kwargs. Right now lm-eval does not pipe the seed correctly to the vLLM request.  
* Default num\_concurrent requests is low (10), so feel free to increase concurrency for better speed.  
* The timeout parameter controls the time in seconds per request. This is optional, but I recommend setting it to a large number for reasoning evals.  
* DOES NOT WORK with log-likelihood tasks (or output\_type: multiple\_choice in lm-eval definitions). Only works for generative tasks.

# lm-eval multiple-choice tasks (no support for chat template)

## Evaluation command

```
lm_eval --model local-completions \
  --tasks mmlu \
  --model_args "model=<model_name>,max_length=<max_length>,base_url=http://0.0.0.0:8000/v1/completions,num_concurrent=10,max_retries=3,tokenized_requests=False" \
  --num_fewshot 5 \
  --output_path results_mmlu.json
```

**Notes:**

* MUST set max\_length explicitly (default=2048 is very low).  
* Default num\_concurrent requests is 1, so feel free to increase concurrency for better speed.

# Useful debugging tips

* Launch limited evaluations for debugging before committing to complete (expensive) evaluations  
  * With lighteval use the argument \--max-samples 5  
  * With lm-eval use the argument \--limit 5  
* Sometimes eval libraries might silently ignore some of your sampling arguments because they are either not parsed correctly or not propagated properly to the actual API call. Before running large scale evaluations, we advise doing a quick debugging run to confirm that all sampling arguments are handled correctly. To do this, start your vLLM server with the following flags:

```shell
VLLM_LOGGING_LEVEL=INFO vllm serve <model> \
    --enable-log-requests \
    --uvicorn-log-level info
```

and verify if your sampling args are correctly reaching the v	LLM server:

```shell
(APIServer pid=953522) INFO 03-20 05:36:49 [logger.py:49] Received request chatcmpl-96ec1b6b48833ca1: params: SamplingParams(n=1, presence_penalty=1.5, frequency_penalty=0.0, repetition_penalty=1.3, temperature=1.0, top_p=0.95, top_k=18, min_p=0.23, seed=42, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=65536, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, structured_outputs=None, extra_args=None)
```

# Standard evaluations

# Standard protocol

* Always use the chat template  
* Use fewshot\_as\_multiturn when using 1 or more fewshot examples  
* Use the vllm serve interface described in the General instructions tab  
* Adjust generation configurations to match the model standard (temperature, top\_p, etc)  
* The list below includes the number of suggested repetitions for each evaluation. Each repetition should use a different random seed  
* For lighteval use **@1@n** to denote average over n repetitions. Example: AIME25 with 8 repetitions: aime25@1@8

# Instruction following

* GSM8k-Platinum  
  * Harness: **lm-eval**  
  * Task: **gsm8k\_platinum\_cot\_llama**  
    * Can be used with models other than Llama too  
  * Number of shots: **5**  
  * Metric: **exact\_match,strict-match**  
  * Repetitions: **3**  
* MMLU-CoT  
  * Harness: **lm-eval**  
  * Task: **mmlu\_cot\_llama**  
  * Number of shots: **5**  
  * Metric: **exact\_match,strict\_match**  
  * Repetitions: **3**  
* MMLU-Pro  
  * Harness: **lm-eval**  
  * Task: **mmlu\_pro\_chat**  
    * Only available on fork at the moment: [https://github.com/neuralmagic/lm-evaluation-harness/tree/mmlu-pro-chat-variant](https://github.com/neuralmagic/lm-evaluation-harness/tree/mmlu-pro-chat-variant)  
  * Number of shots: **5**  
  * Metric: **exact\_match,custom-extract**  
  * Repetitions: **3**  
* IFEval  
  * Harness: **lm-eval**  
  * Task: **ifeval**  
  * Number of shots: **0**  
  * Metric: **inst\_level\_strict\_acc,none**  
  * Repetitions: **3**  
* Math 500  
  * Harness: **lighteval**  
  * Task: **math\_500**  
  * Number of shots: **0**  
  * Metric:  
    * 3 individual jobs: **pass@k:k=1\&n=1**  
    * 1 single job: **pass@k:k=1\&n=3**  
  * Repetitions: **3**  
    * Either 3 individual jobs with different seeds or 1 single job with k=1, n=3

# Reasoning

* GSM8k-Platinum  
  * Harness: **lm-eval**  
  * Task: **gsm8k\_platinum\_cot\_llama**  
    * Can be used with models other than Llama too  
  * Number of shots: **0**  
  * Metric: **exact\_match,strict-match**  
  * Repetitions: **3**  
* MMLU-Pro  
  * lHarness: **lm-eval**  
  * Task: **mmlu\_pro\_chat**  
    * Only available on fork at the moment: [https://github.com/neuralmagic/lm-evaluation-harness/tree/mmlu-pro-chat-variant](https://github.com/neuralmagic/lm-evaluation-harness/tree/mmlu-pro-chat-variant)  
  * Number of shots: **0**  
  * Metric: **exact\_match,custom-extract**  
  * Repetitions: **3**  
* IFEval  
  * Harness: **lm-eval**  
  * Task: **ifeval**  
  * Number of shots: **0**  
  * Metric: **inst\_level\_strict\_acc,none**  
  * Repetitions: **3**  
* Math 500  
  * Harness: **lighteval**  
  * Task: **math\_500**  
  * Number of shots: **0**  
  * Metric:  
    * 3 individual jobs: **pass@k:k=1\&n=1**  
    * 1 single job: **pass@k:k=1\&n=1**  
  * Repetitions: **3**  
    * Either 3 individual jobs with different seeds or 1 single job with k=1, n=3  
* AIME 25  
  * Harness: **lighteval**  
  * Task: **aime25**  
  * Number of shots: **0**  
  * Metric:  
    * 8 individual jobs: **pass@k:k=1\&n=1**  
    * 1 single job: **pass@k:k=1\&n=8**  
  * Repetitions: **8**  
    * Either 8 individual jobs with different seeds or 1 single job with k=1, n=8  
* GPQA Diamond  
  * Harness: **lighteval**  
  * Task: **gpqa:diamond**  
  * Number of shots: **0**  
  * Metric:  
    * 3 individual jobs: **gpqa\_pass@k:k=1\&n=1**  
    * 1 single job: **gpqa\_pass@k:k=1\&n=3**  
  * Repetitions: **3**  
    * Either 3 individual jobs with different seeds or 1 single job with k=1, n=3

# Coding

* LiveCodeBench v6  
  * Harness: **lighteval**  
  * Task: **lcb:codegeneration\_v6**  
  * Number of shots: **0**  
  * Metric:  
    * 3 individual jobs: **codegen\_pass@k:k=1\&n=1**  
    * 1 single job: **codegen\_pass@k:k=1\&n=3**  
  * Repetitions: **3**  
    * Either 3 individual jobs with different seeds or 1 single job with k=1, n=3  
* SWE-Bench  
  * swe-bench \+ mini-swe-agent  
  * Lite  
  * Will share more information about how to use it in coming days

# Long context

* MRCR  
  * Custom evaluation harness  
  * Will add more information in coming days