# ToolMaker Benchmark

This directory contains the task definitions and data for the ToolMaker benchmark.

This folder organized as follows:
- [`tasks/`](tasks/): contains the task definitions
- [`data/`](data/): contains the data for the benchmark
- [`tests/`](tests/): contains the unit tests for the benchmark
- [`papers.yaml`](papers.yaml): contains the papers used in the benchmark
- [`papers.bib`](papers.bib): contains the bibtex entries for the papers

## Task definitions ([`tasks/`](tasks/))

Tasks are defined in the [`tasks/`](tasks/) directory. Each task is defined in a separate YAML file, e.g. the task `medsam_inference` is defined in the file [`tasks/medsam_inference.yaml`](tasks/medsam_train_model.yaml).

Tasks are defined using the following structure:

```yaml
name: stamp_train_classification_model  # name of the task, must correspond to the filename
repo:
  name: STAMP
  url: "https://github.com/KatherLab/STAMP"
  branch: v1
  env: # allows you to specify environment variables that are needed to install the repository or execute the task
    HF_TOKEN: ${env:HF_TOKEN}
papers: [elnahhas2024stamp]  # list of papers that are relevant to the task (must correspond to paper(s) defined in `metadata.yaml`)
category: pathology  # category of the task, e.g. "pathology", "radiology", etc.
description: >-  # description of the task for the LLM
  Train a model for biomarker classification. You will be supplied with the path to the folder containing the whole slide images, alongside a path to a CSV file containing the training labels.
arguments:  # keyword arguments for the task
  slide_dir:  # name of the argument
    description:  Path to the folder containing the whole slide images.  # description of the argument for the LLM
    type: str  # type of the argument, e.g. str, int, bool, etc.
  clini_table:
    description: Path to the CSV file containing the clinical data.
    type: str
  slide_table:
    description: Path to the CSV file containing the slide metadata.
    type: str
  target_column:
    description: The name of the column in the clinical data that contains the target labels.
    type: str
  trained_model_path:
    description: Path to the *.pkl file where the trained model should be saved by this function.
    type: str
returns:  # this section defines the return values of the task, each return value has a name, description and type
  num_params:
    description: The number of parameters in the trained model.
    type: int
example:  # this section defines one example of how to call the task which will be used by the agent to evaluate the proposed code and refine it
  arguments:  # keyword arguments for the example (each argument must correspond to an argument defined in the `arguments` section)
    slide_dir: /mount/input/TCGA-BRCA-SLIDES
    clini_table: /mount/input/TCGA-BRCA-DX_CLINI.xlsx
    slide_table: /mount/input/TCGA-BRCA-DX_SLIDE.csv
    target_column: TP53_driver
    trained_model_path: /mount/output/STAMP-BRCA-TP53-model.pkl
  mount:  # this section defines files/folders to be mounted from the host (in the `benchmark/data` directory) to the container (in the `mount/input`directory) as host:container pairs
    "tcga/TCGA-BRCA-SLIDES": TCGA-BRCA-SLIDES  # the host folder `benchmark/data/tcga/TCGA-BRCA-SLIDES` will be mounted to the container as `/mount/input/TCGA-BRCA-SLIDES`
    "tcga/TCGA-BRCA-DX_CLINI.xlsx": TCGA-BRCA-DX_CLINI.xlsx
    "tcga/TCGA-BRCA-DX_SLIDE.csv": TCGA-BRCA-DX_SLIDE.csv
test_cases:  # this section defines test cases for downstream evaluation
  crc_msi:  # name of the test case
    arguments:  # keyword arguments for the test case (each argument must correspond to an argument defined in the `arguments` section)
      slide_dir: /mount/input/TCGA-CRC-SLIDES
      clini_table: /mount/input/TCGA-BRCA-DX_CLINI.xlsx
      slide_table: /mount/input/TCGA-BRCA-DX_SLIDE.csv
      target_column: TP53_driver
      trained_model_path: /mount/output/STAMP-BRCA-TP53-model.pkl
    mount:  # this section defines files/folders to be mounted from the host (in the `benchmark/data` directory) to the container (in the `mount/input`directory) as host:container pairs
      "tcga/TCGA-CRC-SLIDES": TCGA-CRC-SLIDES  # the host folder `benchmark/data/tcga/TCGA-CRC-SLIDES` will be mounted to the container as `/mount/input/TCGA-CRC-SLIDES`
      "tcga/TCGA-CRC-DX_CLINI.xlsx": TCGA-CRC-DX_CLINI.xlsx
      "tcga/TCGA-CRC-DX_SLIDE.csv": TCGA-CRC-DX_SLIDE.csv
  # more test cases...
    
note: >-  # this note contains additional information about this task, which will NOT be shown to the LLM (it is solely to provide more context to the human reader)
  Some optional note here...
```

## Data ([`data/`](data/))

The data folder contains the data for the benchmark.
Many of the files are large, so they are not included in the repository.
The [`data/download.sh`](data/download.sh) script can be used to download some of the missing files.
The remaining files should be downloaded according to the instructions in the [`data/README.md`](data/README.md) file.

## Papers ([`papers/`](papers/))

The papers folder contains the papers for the benchmark.
Each paper is provided in txt format, e.g. the paper `MedSAM` is provided in the file [`papers/MedSAM.txt`](papers/MedSAM.txt).

## Tests ([`tests/`](tests/))

The tests folder contains the unit tests for the benchmark.
Each tool has a test file, e.g. the test for the tool `uni_extract_features` is defined in the file [`tests/test_uni_extract_features.py`](tests/test_uni_extract_features.py).

At the top of each test file, we call the `initialize()` function:
```python
from tests.utils import initialize

initialize()
```
This function registers a pytest fixture named `tool` for the tool itself, e.g. `uni_extract_features`. It also registers a fixture for each invocation of the tool defined in the YAML file in the `test_cases` section, e.g. `kather100k_muc`.

Loosely speaking, calling the `initialize()` function is equivalent to the following code:
```python
@pytest.fixture(scope="module")
def tool(name: str):
    def wrapper(invocation: ToolInvocation) -> ToolRunResult:
        return run_tool(name, arguments=invocation.arguments, mounts=invocation.mount)

    return wrapper

@pytest.fixture(scope="module")
def kather100k_muc(tool: ToolFixture):
    return tool("kather100k_muc")
```
