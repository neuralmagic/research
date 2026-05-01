**Data Preparation**
* Filtered the arXiv dataset for quantitative finance (`q-fin`) papers and randomly sampled 100 articles.
* Downloaded the source PDFs and extracted them into plain text, preserving logical reading order for double-column formats.

**Model Setup**
* **Engine:** vLLM
* **Base Model:** `Qwen/Qwen3-8B`
* **Draft Model:** `RedHatAI/Qwen3-8B-Thinking-speculator.eagle3`
* **Speculative Config:** Eagle3 method generating 5 speculative tokens per verification step. 
* **Sampling Params:** Temperature 0.6, Top-P 0.95, Top-K 20, Max Output 256 tokens.

**Benchmarking Process**
* Swept across 40 input chunk sizes, scaling from 500 to 20,000 tokens in 500-token increments.
* For each chunk size, I extracted a random text block of that length from all articles that had a number of tokens >= the chunk size.
* Ran a standard summarization prompt **"Generate a concise summary of the following article chunk: [chunk]"** across the batch with "Thinking" turned on.

**Metrics Tracked**
Captured vLLM's internal counter deltas before and after each batch to measure:
* Total draft generation steps.
* Total draft tokens proposed vs. successfully accepted.
* Mean acceptance length.
* Acceptance rate per individual token position (positions 1 through 5).
