import pathlib

STANDARD_CONFIGS = {}
CURRENT_DIR = pathlib.Path(__file__).parent

for file_path in pathlib.Path(CURRENT_DIR).rglob("*.yaml"):
    STANDARD_CONFIGS[file_path.stem] = file_path.resolve()