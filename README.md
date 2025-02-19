# ToolMaker

## Installation
First, install [`uv`](https://github.com/astral-sh/uv)
```bash
pip install uv
```

Then, create and activate a virtual environment with:
```bash
uv sync
source .venv/bin/activate
```

Also, create a `.env` file in the root directory with the following content:
```bash
OPENAI_API_KEY=sk-proj-...  # your OpenAI API key (required to run toolmaker)
HF_TOKEN=hf_...  # your Hugging Face API key (required for some benchmark tools)
CUDA_VISIBLE_DEVICES=0  # if you have a GPU
```

## Usage
First, use toolmaker to install the repository:
```bash
uv run python -m toolmaker install UNI --name my_uni_installed
```

Then, use toolmaker to create the tool:
```bash
uv run python -m toolmaker create uni_extract_features --name my_uni_tool --installed my_uni_installed
```

Finally, you can run the tool on one of the test cases:
```
uv run python -m toolmaker run my_uni_tool --name kather100k_muc
```
Here, `kather100k_muc` is the name of the test case defined in the [tool definition file](benchmark/tasks/uni_extract_features.yaml). 
See [`benchmark/README.md`](benchmark/README.md) for details on how tools are defined.

## Benchmarking
To run the unit tests that constitute the benchmark, use the following command (note that this requires the `benchmark` dependency group to be installed via `uv sync --group benchmark`):
```bash
uv run python -m pytest benchmark/tests --junit-xml=benchmark.xml -m cached  # only run cached tests (faster)
```
This will create a `benchmark.xml` containing JUnit-style XML test results.

## Unit tests
To run toolmaker's own unit tests (not to be confused with the unit tests in the benchmark), use the following command:
```bash
uv run python -m pytest tests
```
