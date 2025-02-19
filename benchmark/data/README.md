# Data

This folder contains the data for the benchmark.
However, large files are not included in the repository.
The [`download.sh`](download.sh) script can be used to download most of the missing files.

The remaining files should be downloaded according to the instructions below:
- `tcga/TCGA-BRCA-SLIDES`: Download the TCGA-BRCA dataset from [GDC](https://portal.gdc.cancer.gov/projects/TCGA-BRCA) and put it in the `tcga/TCGA-BRCA-SLIDES` folder.
- `tcga/TCGA-CRC-SLIDES`: Download the TCGA-CRC dataset from [GDC](https://portal.gdc.cancer.gov/projects/TCGA-CRC) and put it in the `tcga/TCGA-CRC-SLIDES` folder.

## Folder structure
The [`folder_structure.txt`](folder_structure.txt) file shows the folder stucture of the complete data folder (produced by running `find . | sort > folder_structure.txt` in the data folder).
