import argparse
import datetime
import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import fitz
import kagglehub
import requests
from pydantic import BaseModel, Field
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

random.seed(42)

METADATA_FILENAME = "arxiv-metadata-oai-snapshot.json"
Q_FIN_OUTPUT_FILENAME = "q-fin.jsonl"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
PDF_CACHE = "cache/pdfs"
TEXT_CACHE = "cache/text"
MODEL_NAME = "Qwen/Qwen3-8B"
NUM_SPEC_TOKENS = 5
SPECULATIVE_CONFIG = {
    "model": "RedHatAI/Qwen3-8B-Thinking-speculator.eagle3",
    "num_speculative_tokens": NUM_SPEC_TOKENS,
    "method": "eagle3",
}
SAMPLING_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "max_tokens": 256,
}
TOKEN_SWEEP = [500 * (n + 1) for n in range(20)]
BENCHMARK_OUTPUT_DIR = "results/"

parser = argparse.ArgumentParser(description="Script to process articles.")

# Argparse boilerplate for --num-random-articles
parser.add_argument(
    "--num-random-articles",
    type=int,
    default=100,
    help="Number of random articles to process (default: 100)",
)

args = parser.parse_args()

NUM_RANDOM_ARTICLES = args.num_random_articles


class ArxivVersion(BaseModel):
    version: str
    created: str


class ArxivPaper(BaseModel):
    id: str
    submitter: Optional[str] = None
    authors: str
    title: str
    comments: Optional[str] = None
    journal_ref: Optional[str] = Field(None, alias="journal-ref")
    doi: Optional[str] = None
    report_no: Optional[str] = Field(None, alias="report-no")
    categories: str
    license: Optional[str] = None
    abstract: str
    versions: list[ArxivVersion]
    update_date: datetime.date
    authors_parsed: list[list[str]]
    plain_text: str = ""

    model_config = {"populate_by_name": True}


def pdf_filename(paper: ArxivPaper) -> str:
    return paper.id.replace("/", "-") + ".pdf"


def text_filename(paper: ArxivPaper) -> str:
    return paper.id.replace("/", "-") + ".txt"


def download_pdf(paper: ArxivPaper, out_dir: Path = Path(PDF_CACHE)) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(pdf_filename(paper))
    if Path(dest).is_file():
        return True
    url = ARXIV_PDF_URL.format(arxiv_id=paper.id)
    try:
        r = requests.get(
            url,
            stream=True,
            timeout=120,
            headers={"User-Agent": "arxiv-finance-dl/1.0"},
        )
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
        return True
    except Exception:
        return False


def extract_pdf_text(file_path: os.PathLike) -> str | None:
    """
    Extracts plain text from a locally saved arXiv PDF file.

    Args:
        file_path (str): The path to the local PDF file.

    Returns:
        str: The extracted plain text from the PDF.
    """
    # 1. Verify the file exists
    if not os.path.exists(file_path):
        logging.error(f"Error: File not found at '{file_path}'")
        return None

    text_content = []

    try:
        # 2. Open the local PDF document
        with fitz.open(file_path) as doc:
            for page in doc:
                # 'sort=True' tells PyMuPDF to attempt logical reading order,
                # which is crucial for double-column arXiv papers.
                page_text = page.get_text("text", sort=True)
                text_content.append(page_text)

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

    # 3. Join all pages with a clear separator
    return "\n--- PAGE BREAK ---\n".join(text_content)


def count_lines(filename):
    with open(filename, "rb") as f:
        lines = 0
        buf_size = 1024 * 1024  # 1MB
        read_f = f.read
        buf = read_f(buf_size)
        while buf:
            lines += buf.count(b"\n")
            buf = read_f(buf_size)
        return lines


def get_raw_metrics(llm, num_spec_tokens):
    """Fetches a snapshot of the raw cumulative speculative metrics."""
    metrics = llm.get_metrics()
    stats = {
        "num_drafts": 0,
        "num_draft_tokens": 0,
        "num_accepted_tokens": 0,
        "acceptance_counts": [0] * num_spec_tokens,
    }

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts" and isinstance(metric, Counter):
            stats["num_drafts"] += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens" and isinstance(
            metric, Counter
        ):
            stats["num_draft_tokens"] += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens" and isinstance(
            metric, Counter
        ):
            stats["num_accepted_tokens"] += metric.value
        elif (
            metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos"
            and isinstance(metric, Vector)
        ):
            for pos in range(min(len(metric.values), len(stats["acceptance_counts"]))):
                stats["acceptance_counts"][pos] += metric.values[pos]

    return stats


def calculate_metrics_delta(current_stats, previous_stats):
    """Calculates the difference between two metric snapshots."""
    return {
        "num_drafts": current_stats["num_drafts"] - previous_stats["num_drafts"],
        "num_draft_tokens": current_stats["num_draft_tokens"]
        - previous_stats["num_draft_tokens"],
        "num_accepted_tokens": current_stats["num_accepted_tokens"]
        - previous_stats["num_accepted_tokens"],
        "acceptance_counts": [
            curr - prev
            for curr, prev in zip(
                current_stats["acceptance_counts"], previous_stats["acceptance_counts"]
            )
        ],
    }


def compute_speculative_stats(stats, label):
    """Formats, prints, and returns the speculative metrics as a dictionary."""
    num_drafts = stats["num_drafts"]
    num_draft_tokens = stats["num_draft_tokens"]
    num_accepted_tokens = stats["num_accepted_tokens"]
    acceptance_counts = stats["acceptance_counts"]

    # Initialize the dictionary with the base metrics
    results_dict = {
        "label": label,
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
    }

    logging.info("-" * 50)
    logging.info(f"=== {label} SPECULATIVE METRICS ===")
    logging.info(f"num_drafts: {num_drafts}")
    logging.info(f"num_draft_tokens: {num_draft_tokens}")
    logging.info(f"num_accepted_tokens: {num_accepted_tokens}")

    # Calculate and store the mean acceptance length
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    results_dict["mean_acceptance_length"] = acceptance_length

    logging.info(f"mean acceptance length: {acceptance_length:.2f}")
    logging.info("-" * 50)

    # Calculate and store the acceptance rate for each token index
    acceptance_rates = []
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        acceptance_rates.append(acceptance_rate)
        logging.info(f"acceptance at token {i}: {acceptance_rate:.2f}")

    results_dict["acceptance_rates"] = acceptance_rates
    logging.info("-" * 50)

    return results_dict


def get_random_window(lst, window_length):
    """
    Selects a random window of a specified length from a list.
    Returns the whole list if it is shorter than the window length.
    """
    # If the list is shorter than or equal to the window length, return the whole list
    if len(lst) <= window_length:
        return lst

    # Calculate the maximum possible starting index to ensure the window fits
    max_start_index = len(lst) - window_length

    # Choose a random starting index
    start_index = random.randint(0, max_start_index)

    # Return the sliced window
    return lst[start_index : start_index + window_length]


def generate_messages(
    tokenizer, tokenized_articles: list[list[int]], num_tokens: int
) -> list[list[dict[str, str]]]:
    tokenized_article_chunks = [
        get_random_window(t, num_tokens) for t in tokenized_articles
    ]
    article_chunks = [tokenizer.decode(t) for t in tokenized_article_chunks]

    data = [
        [
            {
                "role": "user",
                "content": f"Generate a concise summary of the following article chunk: {chunk}",
            }
        ]
        for chunk in article_chunks
    ]
    return data


def benchmark(
    llm: LLM, messages: list[list[dict[str, str]]], sampling_params: SamplingParams
):
    metrics_before = get_raw_metrics(llm, NUM_SPEC_TOKENS)
    llm.chat(messages, sampling_params=sampling_params, use_tqdm=True, chat_template_kwargs={"enable_thinking": True})  # type: ignore
    metrics_after = get_raw_metrics(llm, NUM_SPEC_TOKENS)
    delta = calculate_metrics_delta(metrics_after, metrics_before)
    stats = compute_speculative_stats(delta, "BENCHMARK")
    return stats


# Download latest version
metadata_dir = kagglehub.dataset_download("Cornell-University/arxiv")
metadata_json_file = Path(metadata_dir) / Path(METADATA_FILENAME)
logging.info(f"ArXiV metadata downloaded to {metadata_json_file}")
num_articles = count_lines(metadata_json_file)
logging.info(f"Metadata contains {num_articles} articles")

q_fin_articles = []
if not Path(Q_FIN_OUTPUT_FILENAME).is_file():
    logging.info("Filtering for q-fin articles")
    with open(metadata_json_file, "r") as file:
        for line in tqdm(file, total=num_articles):
            # Process the line (e.g., print or strip whitespace)
            article_dict = json.loads(line)
            article = ArxivPaper.model_validate(article_dict)
            if "q-fin" in article.categories:
                q_fin_articles.append(article)

    num_qfin_articles = len(q_fin_articles)
    logging.info(f"{num_qfin_articles} articles found")
    with open(Q_FIN_OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        for article in q_fin_articles:
            # model_dump_json() returns a JSON string
            f.write(article.model_dump_json() + "\n")
else:
    logging.info(
        f"{Q_FIN_OUTPUT_FILENAME} file already exists. Skipping article metadata filtering."
    )
    logging.info(f"Reading q-fin articles from {Q_FIN_OUTPUT_FILENAME}")
    num_articles = count_lines(Q_FIN_OUTPUT_FILENAME)
    with open(Q_FIN_OUTPUT_FILENAME, "r") as file:
        for line in tqdm(file, total=num_articles):
            # Process the line (e.g., print or strip whitespace)
            article_dict = json.loads(line)
            article = ArxivPaper.model_validate(article_dict)
            q_fin_articles.append(article)

logging.info(f"Selecting {NUM_RANDOM_ARTICLES} random articles")
q_fin_filtered: list[ArxivPaper] = random.choices(q_fin_articles, k=NUM_RANDOM_ARTICLES)
logging.info(f"Downloading articles to {PDF_CACHE}")
for article in tqdm(q_fin_filtered):
    download_pdf(article)

logging.info(f"Extracting article content to {TEXT_CACHE}")
Path(TEXT_CACHE).mkdir(parents=True, exist_ok=True)
articles: list[ArxivPaper] = []
for article in tqdm(q_fin_filtered):
    out_path = Path(TEXT_CACHE) / Path(text_filename(article))
    if out_path.is_file():
        with open(out_path, "r") as f:
            article.plain_text = f.read()
            articles.append(article)
        continue
    in_path = Path(PDF_CACHE) / Path(pdf_filename(article))
    article_text = extract_pdf_text(in_path)
    if article_text is None:
        continue
    articles.append(article)
    with open(str(out_path), "w") as f:
        f.write(article_text)

logging.info(f"Loading {MODEL_NAME} in vLLM")
sampling_params = SamplingParams(**SAMPLING_PARAMS)
llm = LLM(
    model="Qwen/Qwen3-8B",
    trust_remote_code=True,
    disable_log_stats=False,
    speculative_config=SPECULATIVE_CONFIG,
)

logging.info(f"Tokenizing Articles")
tokenizer = llm.get_tokenizer()
tokenized = tokenizer.encode([a.plain_text for a in articles])
sizes_output = {"article_sizes": [len(t) for t in tokenized]}
with open("article_sizes.json", "w") as f:
    json.dump(sizes_output, f)
logging.info(f"Saving results to {BENCHMARK_OUTPUT_DIR}")
Path(BENCHMARK_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
for num_tokens in TOKEN_SWEEP:
    output_filename = Path(BENCHMARK_OUTPUT_DIR) / Path(f"chunk_size_{num_tokens}.json")
    logging.info(f"Benchmarking with {num_tokens}-sized chunks")
    data = generate_messages(tokenizer, tokenized, num_tokens)
    result = benchmark(llm, data, sampling_params)
    result["chunk_size"] = num_tokens
    with open(str(output_filename), "w") as f:
        json.dump(result, f)
