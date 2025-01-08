## Deep Learning Analysis on Images of iPSC-derived Motor Neurons Carrying fALS-genetics Reveals Disease-Relevant Phenotypes




### Installation
This project uses [pixi](https://prefix.dev/) to install and manage dependencies.  To install pixi, run:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

You can then set up the project using the `setup` make target.
```bash
make setup
```

### Overview
This repo contains the code to reproduce the figures from the paper "Deep Learning Analysis on Images of iPSC-derived Motor Neurons Carrying fALS-genetics Reveals Disease-Relevant Phenotypes".  The code is organized into the following directories:
* src/fals: contains utility functions and scripts for data loading, preprocessing, and visualization
* notebooks: stand-alone notebooks for reproducing the figures in the paper

### Usage
To run JupyterLab and explore notebooks:
```bash
make notebook
```

### Data
Data to reproduce the figures from the paper is available from AWS s3 in the `s3://2025-fals` bucket.  The [config file](src/fals/config.py) refers to this path and can be overridden by setting the `ROOT_DATA_PATH` environment variable if you chose to download the data locally.

### Figures
Below is a summary of the key figures reproduces by the notebooks, illustrating important findings on phenotypic
differences between mutuant and wild-type cells lines via statistical analysis, ML classification, and RNA imputation.

Figure 2
* [2.E](notebooks/anchor_phenotype_figures.ipynb): dose-response curves of TDP-43 mislocalization in four stressor conditions and three seeding densities
* [2.E, F (supp.)](notebooks/anchor_phenotype_figures.ipynb): STMN2 intensity within the soma and neurites

Figure 3
* [3.A](notebooks/anchor_phenotype_figures.ipynb): cell density vs. TDP-43 mislocalization correlation
* [3.B](notebooks/anchor_phenotype_figures.ipynb): TDP-43 mislocalization mask ratio coefficients using a linear model accounting for density across all wells
* [3.D](notebooks/anchor_phenotype_figures.ipynb): WT/VCP-R115C het regression coefficients
* [3.F](notebooks/anchor_phenotype_figures.ipynb): C9ORF72/corrected regression coefficients
* [3.A (supp.)](notebooks/anchor_phenotype_figures.ipynb): density matching-based well replicates selection schematic
* [3.B (supp.)](notebooks/anchor_phenotype_figures.ipynb): TDP-43 mislocalization heatmap with density-matched comparisons
* [3.D (supp.)](notebooks/anchor_phenotype_figures.ipynb): STMN2 intensity vs. cell density
* [3.E (supp.)](notebooks/anchor_phenotype_figures.ipynb): STMN2 intensity coefficients heatmap from a linear model accounting for donor and live cell density
* [3.F, G (supp.)](notebooks/anchor_phenotype_figures.ipynb): soma and neurite STMN2 expression intensity under basal conditions


Figure 4
* [4.A](notebooks/ml_imaging_figures.ipynb): correlation between predicted and actual TDP-43 mislocalization values derived from morphology embeddings
* [4.D, E](notebooks/ml_rna_imputation_figures.ipynb): measured vs. imputed change of expression for ALS genes and all genes
* [4.A (supp.)](notebooks/ml_imaging_figures.ipynb): density histogram of the TDP-43 localization as predicted from DAPI and TUJ1
* [4.D (supp.)](notebooks/ml_rna_imputation_figures.ipynb): RNA imputation from morphology-derived embeddings of each familial ALS mutant

Figure 5
* [5.A](notebooks/ml_imaging_figures.ipynb): classifier accuracy differentiating mutant from WT cells using different feature sets
* [5.C](notebooks/ml_imaging_figures.ipynb): mean accuracy across mutants in a donor hold-out regime with different feature combinations
* [5.E](notebooks/ml_imaging_figures.ipynb): classifier accuracy using the TDP-43 mislocalization ratio versus all features
* [5.F](notebooks/ml_imaging_figures.ipynb): power analysis of phenotypic reversion in a simulated C9ORF72 mutant screen
