# dataset-composition-for-genomic-islands-detection
This is a repository to reproduce the results for the following paper titled "The Contribution of Taxonomic Diversity to Machine Learning Performance for the Detection of Genomic Islands".

**Abstract**:

Genomic islands (GIs) are the main vehicles for horizontal gene transfer (HGT), which facilitates the rapid dissemination of antimicrobial resistance and virulence factors. 
Precise GI detection is essential for tracking pathogen evolution and controlling outbreaks. 
HGT is an inherently multi-species process.
Although other genomic tasks have leveraged cross-species machine learning (ML) models to expand analysis to understudied species, cross-species models for GI detection are lacking.
To this end, we examined how dataset composition impacts cross-species generalizability for GI detection. 
Our results demonstrate that species richness in the training dataset and taxonomic distance to the test species are important for cross-species generalization.
A negative Pearson correlation ($r=-0.74$, $R^2=0.55$, $p<0.001$) between model performance and taxonomic distance, as measured by Jensen-Shannon divergence (JSD), suggests that greater taxonomic distance diminishes cross-species generalization.
This study provides insight into future developments in ML approaches for GI detection and emphasizes the importance of quantifying cross-species generalization when reporting model performance.

---

**Initial steps:**

1. Clone this repo:

```
git clone git@github.com:andrejw27/dataset-composition-for-genomic-islands-detection.git
```

2. `cd` into the root directory (`dataset-composition-for-genomic-islands-detection/`)
3. Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html): `conda/install.sh`
4. Create conda environment: `sh conda/create_env.sh`
5. Activate conda environment: `conda activate datacompos`
6. **Optional**: Remove conda environment (if necessary): `sh conda/remove_env.sh`
7. Extract all files with the form tar.gz (in train_folder and test_folder).
8. Put all the extracted files in the test_folder inside the test_folder (e.g., `./test_folder/in_species.fasta`, `./test_folder/out_species.fasta`, etc.)

---
**Create data sets**:
* run `./0_create_training_pool.py` to create training data pool.

```
python 0_create_training_pool.py
```

* This involves the following steps:
  - download genomic islands predictions from [IslandViewer4 database](https://www.pathogenomics.sfu.ca/islandviewer/download/datasets/)
  - download accession2taxid from https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/accession2taxid/nucl_gb.accession2taxid.gz to assign taxonomic information to each accession
  - sample some species (from [WHO](https://www.who.int/news/item/17-05-2024-who-updates-list-of-drug-resistant-bacteria-most-threatening-to-human-health) and [RKI](https://www.rki.de/EN/Institute/Organisation/Departments/Department-3/Unit-37/Downloads/Pathogen_list_and_criteria_reserve_antibiotics.pdf?__blob=publicationFile&v=8) pathogen lists)
  
* run `./0_deduplicate_data.py` to remove high-similar data with a similarity threshold of 95% (in our paper)
This might take some time, but you can find the deduplicated data in `dataset/train_folder`.

```
python 0_deduplicate_data.py
```

* run `./0_create_training_data.py` to create training data sets

* running this script will produce the following folders:
  - `dataset/train_folder/data` consists of training data sets of different cases (case1, case2a, case2b, and case3)
  - `dataset/train_folder/folds` contains pre-defined folds of each training data set
  - `dataset/train_folder/info` stores information of each training data set, which species are sampled in each training data set
  - `dataset/train_folder/training_pool` comprises training data pool used for training/cross-validation

```
python 0_create_training_data.py
```

**Cross-validation**:

* run `./1_run_cv.py` which requires 4 arguments:
- **representation-index**: to select which data representations to be used in cross-validation
- **n_worker**: number of workers to execute the script for parallel processing
- **filename**: filename of the training data set (case1_1, case1_2, etc.)
- **train-folder**: folder of the training data sets

```
python 1_run_cv.py --representation-index 1 --n-worker 5 --filename "deduplicated_case1_1" --train-folder "dataset/train_folder"
```

* run `./1_compute_cv.py` to read predictions from the cross-validation and compute the evaluation metrics (F1 score, Precision, Recall, Matthews Correlation Coefficient (MCC), and Accuracy)

```
python 1_compute_cv.py --filename "deduplicated_case1_1" --data-folder "dataset/train_folder" --output-folder "outputs/crossval" --predictions-folder "outputs/predictions/deduplicated_samples"
```

**Hyperparameter tuning**:

* run  `./2_hpo.py` to perform a grid search to find the best hyperparameters of the best-performing model and representation (SVM and RCKmer-7)

```
python 2_hpo.py 
```

**Compute Jensen-Shannon Divergence (JSD)**:
* we measure the taxonomic distance between each training data set and each test data set using JSD.
* run  `./3_compute_jsd.py` to compute the divergence between each training data set and each test data set

```
python 3_compute_jsd.py 
```

---

To reproduce tables and figures in the manuscript, we provide a jupyter notebook **visualization.ipynb**
