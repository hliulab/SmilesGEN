# SmilesGEN
## Introduction
We propose SmilesGEN, a model that generates drugs based on gene expression profiles. This model can be used to generate and optimize drug-like molecules using the desired gene expression profile. The following is a detailed introduction to the model:
## Model Architecture
![](https://anonymous.4open.science/r/SmilesGEN/framework.png)


## Environment Installation
The required packages are listed in the requirements.txt. 
User can execute the following command to install the packages:

```
$ pip install -r requirements.txt
```

## File Description

- **datasets**: This file stores the training and testing data.

- **model**: The specific implementation of the SmilesGEN model stores here.

- **results**: The training results store here.

- dataset.py: Code used for dataset preprocessing.

- evaluation.py: Code used to evaluate the model.

- generation.py: Code used to generate drug-like molecules.

- main.py: Model parameters, pre-training, training, testing, and validation code.

- requirements.txt: The environment of the model.

- MolecularOptimization.py: Code for molecular optimization using trained model.

- tokenizer.py: SMILES encoding.

- trainer.py: Training source code.

- utils.py: Other tool codes.
- We collect the drug-treated and untreated expression profiles from the L1000 dataset (https://clue.io/data/CMap2020#LINCS2020). Obtaining molecular SMILES from https://pubchem.ncbi.nlm.nih.gov/. Obtaining known ligands from DTC (https://drugtargetcommons.fimm.fi/). 

## Experimental Reproduction

  - **STEP 1**: Train

  ```
$ python main.py --pre_train_smiles_vae --train --cell_name=MCF7
  ```

  - **STEP 2**: Test

  ```
$ python main.py --use_seed --generation --cell_name=MCF7 --protein_name=AKT1
  ```

  - **STEP 3**: Evaluate

  ```
$ python main.py --use_seed --generation --cell_name=MCF7 --protein_name=AKT1
  ```

  - **STEP 4**: Molecular Optimization

  ```
$ python MolecularOptimization.py --cell_name=MCF7 --protein_name=AKT1
  ```

