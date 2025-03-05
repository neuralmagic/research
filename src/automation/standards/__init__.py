# from automation.standards.benchmarking.benchmarking_code_completion import BenchmarkingCodeCompletionTask
# from automation.standards.benchmarking.benchmarking_chat import BenchmarkingChatTask
# from automation.standards.benchmarking.benchmarking_instruction import BenchmarkingInstructionTask
# from automation.standards.benchmarking.benchmarking_docstring_generation import BenchmarkingDocstringGenerationTask
# from automation.standards.benchmarking.benchmarking_summarization import BenchmarkingSummarizationTask
# from automation.standards.benchmarking.benchmarking_long_rag import BenchmarkingLongRAGTask
# from automation.standards.benchmarking.benchmarking_rag import BenchmarkingRAGTask
# from automation.standards.benchmarking.benchmarking_code_fixing import BenchmarkingCodeFixingTask
# from automation.standards.benchmarking.benchmarking_16k import Benchmarking16kTask
# from automation.standards.benchmarking.benchmarking_32k import Benchmarking32kTask
# from automation.standards.benchmarking.benchmarking_64k import Benchmarking64kTask
# from automation.standards.benchmarking.benchmarking_128k import Benchmarking128kTask
# from automation.standards.evaluations.openllm import OpenLLMTask
# from automation.standards.evaluations.leaderboard import LeaderboardTask
# from automation.standards.compression.quantization_w4a16 import QuantizationW4A16Task
# from automation.standards.compression.quantization_w8a8 import QuantizationW8A8Task
# from automation.standards.compression.quantization_fp8_dynamic import QuantizationFP8DynamicTask
# from automation.standards.compression.pipeline_w4a16 import QuantizationW4A16Pipeline
# from automation.standards.compression.pipeline_w8a8 import QuantizationW8A8Pipeline

import pathlib

STANDARD_CONFIGS = {}
CURRENT_DIR = pathlib.Path(__file__).parent

for file_path in pathlib.Path(CURRENT_DIR).rglob("*.yaml"):
    STANDARD_CONFIGS[file_path.stem] = file_path.resolve()