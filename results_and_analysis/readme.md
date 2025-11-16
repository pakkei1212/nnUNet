# Segmentation Model Evaluation Results

This directory contains the results and analysis of various segmentation models, comparing their performance using the Dice Similarity Coefficient (DSC).

## Files

- `Evaluation.ipynb`: A Jupyter Notebook that performs the analysis. It loads the summary JSON files, extracts DSC values, and generates comparison plots.
- `DA5_CEDice_summary.json`: Summary results for the nnUNetTrainerDA5-CE+Dice setting.
- `DA5_FocalDice_summary.json`: Summary results for the nnUNetTrainerDA5-Focal+Dice setting.
- `Default_CEDice_summary.json`: Summary results for the nnUNetTrainer-CE+Dice setting.
- `Default_FocalDice_summary.json`: Summary results for the nnUNetTrainer-Focal+Dice setting.
- `NoMirroring_CEDice_summary.json`: Summary results for the nnUNetTrainerNoMirroring-CE+Dice setting.
- `NoMirroring_FocalDice_summary.json`: Summary results for the nnUNetTrainerNoMirroring-Focal+Dice setting.
- `mean_dsc_comparison.png`: Bar chart comparing the mean DSC across all evaluated settings.
- `organ_dsc_comparison.png`: Grouped bar chart comparing the per-organ DSC across all evaluated settings.

## Analysis

The `Evaluation.ipynb` notebook performs the following steps:
1. Loads all `_summary.json` files, which contain detailed segmentation metrics for different model configurations.
2. Extracts the mean Dice Similarity Coefficient (DSC) for each setting.
3. Generates a bar plot (`mean_dsc_comparison.png`) to visualize the overall mean DSC for each setting.
4. Extracts per-organ DSC values for each setting.
5. Generates a grouped bar plot (`organ_dsc_comparison.png`) to visualize the DSC for each organ across different settings.

## How to Use

To reproduce the analysis or view the plots:
1. Ensure you have Python and Jupyter Notebook installed.
2. Install the required libraries: `pandas`, `matplotlib`, `numpy`.
3. Open `Evaluation.ipynb` in Jupyter Notebook.
4. Run all cells to regenerate the plots and view the dataframes.