# thermal-comfort-prediction
## Project Overview

This project aims to analyze and predict thermal comfort based on various environmental and personal factors. The dataset used includes information on air temperature, humidity, clothing insulation, metabolic rate, and other relevant variables.

<!-- ## Folder Structure

- `code/`
  - `sampled_ashrae_data.csv`: Contains sampled data from the ASHRAE database.
  - `processed_sampled_ashrae_data.csv`: Processed version of the sampled data.
  - `fe_sampled_simplified_ashrae_data.csv`: Feature-engineered and simplified version of the sampled data.
  - `thermal_comfort_analysis.py`: Python script for analyzing thermal comfort.
  - `thermal_comfort_analysis.ipynb`: Jupyter notebook for interactive analysis and visualization.
  - `2. THERMAL_COMFORT_ANALYSIS_FE_BASE.ipynb`: Jupyter notebook focusing on feature engineering and base analysis. -->

## Data Processing

The data processing steps include:
1. Handling missing values.
2. Dropping irrelevant or sparse columns.
3. Renaming columns for better readability.
4. Capping outliers to ensure data quality.

## Analysis and Modeling

The analysis involves:
- Exploratory Data Analysis (EDA) to understand data patterns and relationships.
- Visualization of data distributions and correlations.
- Building predictive models using machine learning algorithms such as Logistic Regression and Random Forest Classifier.

## Visualization

The project includes various visualizations to aid in understanding the data and model performance:
- Histograms and boxplots for numerical features.
- Correlation heatmaps.
- Scatter plots for key relationships.
- Confusion matrices for model evaluation.

## How to Run

1. Ensure you have the necessary Python libraries installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
2. Run the Python script:
   ```bash
   python thermal_comfort_analysis.py
   ```
3. Alternatively, open the Jupyter notebooks for an interactive analysis:
   ```bash
   jupyter notebook thermal_comfort_analysis.ipynb
   ```

## Contributors

- Sakif Khan

## License

This project is licensed under the MIT License.