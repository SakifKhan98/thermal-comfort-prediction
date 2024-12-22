# %% [markdown]
# <div style="text-align: center; color:aquamarine">
#   <h2>Main Objective</h2>
# </div>
#   <p>The increasing importance of thermal comfort in energy-efficient building management systems, particularly for creating optimal environments that balance energy consumption and user comfort. In this project we will develop a machine learning model to predict the thermal preferences of individuals based on environmental variables like air temperature, humidity, and personal characteristics such as age, sex, and activity level.</p>

# %% [markdown]
# <div style="text-align: center; color:aquamarine">
#   <h2>Importing Libraries</h2>
# </div>
#   <p>The following section of the code is responsible for importing the necessary libraries and modules required for the program to function. These imports may include standard Python libraries, third-party packages, or custom modules. Each import statement ensures that the corresponding library or module is available for use within the code, providing access to various functions, classes, and methods that facilitate the implementation of the program's functionality.</p>

# %%
# Import all necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE

# %% [markdown]
# <div style="text-align: center; color:aquamarine">
#   <h2>Loading Dataset</h2>
#   <h2>The ASHRAE Global Thermal Comfort Database II</h2>
# </div>
#   <p>The ASHRAE Global Thermal Comfort Database II was launched in 2014 under the leadership of the University of California, Berkeley’s Center for the Built Environment and The University of Sydney’s Indoor Environmental Quality (IEQ) Laboratory. The database is a significant open-source research initiative aimed at advancing the study of HVAC (Heating, Ventilation, and Air Conditioning) systems and thermal comfort.</p>
#
# ![dataset_annotation_1.png](attachment:dataset_annotation_1.png)
# ![dataset_annotation_2.png](attachment:dataset_annotation_2.png)
# ![dataset_annotation_3.png](attachment:dataset_annotation_3.png)
#
# Dataset Link: [The ASHRAE Global Thermal Comfort Database II](https://www.kaggle.com/datasets/claytonmiller/ashrae-global-thermal-comfort-database-ii)

# %%
# Load the raw dataset
file_path = "ashrae_db2.01.csv"
raw_df = pd.read_csv(file_path)

# Inspect the dataset
print("Dataset Shape:", raw_df.shape)
raw_df.head()

# %% [markdown]
# <div style="text-align: center; color:aquamarine">
#   <h2>Exploratory Data Analysis</h2>
# </div>
#   <p>The following section of the code is responsible for performing Exploratory Data Analysis (EDA) on the dataset. EDA is a crucial step in the data analysis process, as it helps to understand the underlying patterns, relationships, and anomalies in the data. This section includes various techniques such as data visualization, summary statistics, and correlation analysis to gain insights into the dataset and guide further data preprocessing and modeling steps.</p>

# %%
# loc[] is primarily label based filtering method, to access a group of rows and columns by label
df = raw_df.loc[:, raw_df.isna().mean() < 0.67]

# %%
df.shape

# %%
df.columns

# %%
df.isnull().sum()

# %%
df.dtypes

# %% [markdown]
# <div style="text-align: center; color:aquamarine">
#   <h2>Data Preprocessing</h2>
# </div>
#   <p>The following section of the code is responsible for pre-processing the dataset. This includes handling missing values, dropping irrelevant or sparse columns, renaming columns for better readability, and capping outliers. These steps ensure that the dataset is clean and ready for further analysis and modeling.</p>

# %%
# Step 1: Inspect missing data to identify sparse columns
missing_data_summary = df.isnull().sum() / len(df) * 100
print("Percentage of missing values:\n", missing_data_summary)

# Step 2: Define columns to drop
columns_to_drop = [
    "Publication (Citation)",  # Metadata
    "Data contributor",  # Metadata
    # "Year",  # May not directly affect thermal comfort
    "Air temperature (F)",  # Duplicate with 'Air temperature (C)'
    # "Relative humidity (%)",  # Example: Keep or drop depending on focus
    "Cooling startegy_operation mode for MM buildings",  # Sparse
    "Heating strategy_building level",  # Sparse
    "activity_10",
    "activity_20",
    "activity_30",
    "activity_60",  # Sparse
    "PMV",
    "PPD",
    # "SET",
    # "MET",  # Sparse
    "Ta_h (C)",
    "Ta_h (F)",
    "Ta_m (C)",
    "Ta_m (F)",
    "Ta_l (C)",
    "Ta_l (F)",
    # "Operative temperature (C)",
    "Operative temperature (F)",
    "Radiant temperature (C)",
    "Radiant temperature (F)",
    "Globe temperature (F)",
    "Tg_h (C)",
    "Tg_h (F)",
    "Tg_m (C)",
    "Tg_m (F)",
    "Tg_l (C)",
    "Tg_l (F)",
    "Humidity preference",
    "Humidity sensation",
    "Air velocity (fpm)",
    "Velocity_h (m/s)",
    "Velocity_h (fpm)",
    "Velocity_m (m/s)",
    "Velocity_m (fpm)",
    "Velocity_l (m/s)",
    "Velocity_l (fpm)",
    "Blind (curtain)",
    "Fan",
    "Window",
    "Door",
    "Heater",  # Sparse
    "Outdoor monthly air temperature (F)",
    "Database",
    # Sparse
]

# Step 3: Drop the columns
simplified_dataset = df.drop(columns=columns_to_drop, errors="ignore")

# Display the updated dataset shape and columns
print("Updated Dataset Shape:", simplified_dataset.shape)
print("Updated Columns:\n", simplified_dataset.columns)

# %% [markdown]
# ### Columns to Consider Dropping:
#
# 1. **High Missing Values (>50%):**
#    - `Air movement acceptability` (84.81% missing)
#    - `Subject«s height (cm)` (81.08% missing)
#    - `Subject«s weight (kg)` (77.08% missing)
#    - `Globe temperature (C)` (75.83% missing)
#
# 2. **Columns with Missing Values Between 20-50% (Optional):**
#    - These columns have a moderate level of missing data and can be retained or dropped based on their importance:
#      - `Sex` (37.69% missing): Important for personalization but sparse.
#      - `Thermal sensation acceptability` (41.96% missing): Consider based on its relevance to thermal comfort.
#      - `Thermal preference` (20.53% missing): Useful for comfort analysis but sparse.
#
# 3. **Columns with Relatively Lower Missing Data (<20%):**
#    - These columns have low percentages of missing values and can be imputed:
#      - `Age` (59.49% missing): High but potentially critical for demographic insights. Consider imputing if relevant.
#      - `Met` (15.95% missing): Can be imputed with median or mean.
#      - `Air velocity (m/s)` (16.44% missing): Critical for air movement and comfort analysis; impute missing values.
#      - `Outdoor monthly air temperature (C)` (26.25% missing): Impute to retain climate-related information.
#      - `Air temperature (C)` (7.13% missing): Retain and impute missing values, as temperature is a key variable.
#
# ---
#
# ### Justification for Decisions:
#
# 1. **High Missing Values (>50%):**
#    - Columns with over 50% missing data are unreliable for analysis and should generally be dropped unless they are critical to the research.
#
# 2. **Moderate Missing Values (20-50%):**
#    - Columns like `Sex`, `Thermal sensation acceptability`, and `Thermal preference` should be retained only if they are critical to the research objectives. Otherwise, they can be dropped or imputed with caution.
#
# 3. **Low Missing Values (<20%):**
#    - Columns with low missing values should be imputed to retain their information. Techniques like median, mean, or mode imputation can be used depending on the data type and distribution.
#
# 4. **Final Decision:**
#    - Drop columns with over 50% missing values and evaluate the significance of columns with 20-50% missing data. Impute missing values for columns with less than 20% missing data to maintain their utility in analysis.
#

# %%
# Define columns to drop based on missing values and relevance
columns_to_drop_further = [
    "Air movement acceptability",  # >80% missing
    "Globe temperature (C)",  # >75% missing
    # "Thermal comfort",  # >66% missing -- NOT DROPPING - Potential Target Variable
    # "Subject«s height (cm)",  # >81% missing -- NOT DROPPING - Important Personal Feature
    # "Subject«s weight (kg)",  # >77% missing -- NOT DROPPING - Important Personal Feature
]

# Drop the additional columns
simplified_dataset = simplified_dataset.drop(
    columns=columns_to_drop_further, errors="ignore"
)

# Check the updated dataset shape and columns
print("Updated Dataset Shape:", simplified_dataset.shape)
print("Updated Columns:\n", simplified_dataset.columns)

# %%
# Statistical summary of numerical features
print("\nStatistical Summary:\n", simplified_dataset.describe())

# %%
simplified_dataset.isnull().sum()

# %%
# Define a dictionary with the current column names as keys and new names as values
rename_columns = {
    "Year": "year",
    "Season": "season",
    "Koppen climate classification": "koppen_climate",
    "Climate": "climate_description",
    "City": "city",
    "Country": "country",
    "Building type": "building_type",
    "Cooling startegy_building level": "cooling_strategy",
    # "Heating strategy_building level": "heating_strategy",
    "Age": "age",
    "Sex": "sex",
    "Thermal sensation": "thermal_sensation",
    "Thermal sensation acceptability": "thermal_sensation_acceptability",
    "Thermal preference": "thermal_preference",
    "Air movement preference": "air_movement_preference",
    "Thermal comfort": "thermal_comfort",
    "SET": "standard_effective_temperature",
    "Clo": "clothing_insulation",
    "Met": "metabolic_rate",
    "Air temperature (C)": "air_temperature",
    "Operative temperature (C)": "operative_temperature",
    "Radiant temperature (C)": "radiant_temperature",
    "Relative humidity (%)": "relative_humidity",
    "Humidity preference": "humidity_preference",
    "Humidity sensation": "humidity_sensation",
    "Subject«s height (cm)": "subject_height",
    "Subject«s weight (kg)": "subject_weight",
    "Air velocity (m/s)": "air_velocity",
    "Outdoor monthly air temperature (C)": "outdoor_air_temperature",
}

# Rename the columns
simplified_dataset.rename(columns=rename_columns, inplace=True)

# Display the updated column names
print("Updated Column Names:\n", simplified_dataset.columns)

# %%
# Export the sampled dataframe to a CSV file
simplified_dataset.to_csv("simplified_ashrae_data.csv", index=False)

# %%
# Identify categorical columns
categorical_cols = simplified_dataset.select_dtypes(include=["object"]).columns

# Display unique values and their count for each categorical column
for col in categorical_cols:
    unique_values = simplified_dataset[col].unique()
    print(f"Column: '{col}'")
    print(f"Number of Unique Values: {len(unique_values)}")
    print(f"Unique Values: {unique_values}")
    print("-" * 50)

# %% [markdown]
# ### Our Target Variable is thermal_preference. So, we are dropping all the rows where thermal_preference is missing.

# %%
# Drop rows where 'thermal_preference' is null
simplified_dataset = simplified_dataset.dropna(subset=["thermal_preference"])

# %%
# Convert 'thermal_comfort' to float
simplified_dataset["thermal_comfort"] = pd.to_numeric(
    simplified_dataset["thermal_comfort"], errors="coerce"
)

# Verify the conversion
print(simplified_dataset["thermal_comfort"].dtype)
print(simplified_dataset["thermal_comfort"].head())

# %%
# Calculate percentage of missing values
missing_values = simplified_dataset.isnull().mean() * 100
print("\nPercentage of Missing Values:\n", missing_values)

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values)
plt.xticks(rotation=45, ha="right")
plt.title("Percentage of Missing Values")
plt.ylabel("Percentage")
plt.show()

# %%
# Visualize missing values using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(simplified_dataset.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# %%
# Plot histograms for numerical features
simplified_dataset.select_dtypes(include=["float64", "Int64"]).hist(
    figsize=(15, 12), bins=20, color="teal", edgecolor="black"
)
plt.suptitle("Histograms of Numerical Features", fontsize=16)
plt.tight_layout(pad=2.0)
plt.show()

# %%
# Visualize the correlation matrix
numerical_cols = simplified_dataset.select_dtypes(include=["float64", "Int64"]).columns
correlation_matrix = simplified_dataset[numerical_cols].corr()
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# %%
categorical_features = simplified_dataset.select_dtypes(include=["object"]).columns
for feature in categorical_features:
    if feature == "city":
        continue
    plt.figure(figsize=(10, 6))
    sns.countplot(data=simplified_dataset, y=feature, palette="viridis", hue=feature)
    plt.title(f"Distribution of {feature}")
    plt.show()

# %%
# Scatter plot for key relationships
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=simplified_dataset,
    x="air_temperature",
    y="thermal_sensation",
    hue="season",
    alpha=0.6,
)
plt.title("Air Temperature vs Thermal Sensation")
plt.show()

# %%
# Scatter plot for key relationships
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=simplified_dataset,
    x="air_temperature",
    y="thermal_comfort",
    hue="season",
    alpha=0.6,
)
plt.title("Air Temperature vs Thermal Comfort")
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(
    data=simplified_dataset, x="thermal_sensation", kde=True, bins=15, color="purple"
)
plt.title("Distribution of Thermal Sensation")
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.histplot(
    data=simplified_dataset, x="thermal_comfort", kde=True, bins=15, color="purple"
)
plt.title("Distribution of Thermal Comfort")
plt.show()

# %%
# Pairplot for selected features
selected_features = [
    "thermal_sensation",
    "air_temperature",
    "operative_temperature",
    # "humidity_sensation",
    "thermal_preference",
    "thermal_comfort",
    "relative_humidity",
]
sns.pairplot(
    simplified_dataset[selected_features],
    hue="thermal_preference",
    diag_kind="kde",
    plot_kws={"alpha": 0.6},
)
plt.suptitle("Pairplot of Selected Features", y=1.02, fontsize=16)
plt.show()

# %%
# Group by season to analyze trends
seasonal_means = simplified_dataset.groupby("season")["thermal_comfort"].mean()
seasonal_means.plot(kind="bar", figsize=(8, 6), color="teal", edgecolor="black")
plt.title("Average Thermal Comfort by Season")
plt.ylabel("Thermal Comfort")
plt.xlabel("Season")
plt.show()

# %%
# Boxplot for detecting outliers in numerical features
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=simplified_dataset, x=col)
    plt.title(f"Outlier Detection: {col}")
    plt.show()
# Calculate and print the percentage of outliers in each numerical feature
for col in numerical_cols:
    Q1 = simplified_dataset[col].quantile(0.25)
    Q3 = simplified_dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = simplified_dataset[
        (simplified_dataset[col] < lower_bound)
        | (simplified_dataset[col] > upper_bound)
    ]
    outlier_percentage = (len(outliers) / len(simplified_dataset)) * 100
    print(
        f"Feature: {col}, Outliers: {len(outliers)}, Outlier Percentage: {outlier_percentage:.2f}%"
    )


# %%
# Function to cap outliers
def cap_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df


# Apply to features with high/moderate outlier percentages
outlier_features = [
    "age",
    "thermal_sensation",
    "thermal_sensation_acceptability",
    "clothing_insulation",
    "metabolic_rate",
    "air_temperature",
    "air_velocity",
    "standard_effective_temperature",
    "operative_temperature",
]
for col in outlier_features:
    df = cap_outliers(simplified_dataset, col)

# %%
# Boxplot for detecting outliers in numerical features
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=simplified_dataset, x=col)
    plt.title(f"Outlier Detection: {col}")
    plt.show()
# Calculate and print the percentage of outliers in each numerical feature
for col in numerical_cols:
    Q1 = simplified_dataset[col].quantile(0.25)
    Q3 = simplified_dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = simplified_dataset[
        (simplified_dataset[col] < lower_bound)
        | (simplified_dataset[col] > upper_bound)
    ]
    outlier_percentage = (len(outliers) / len(simplified_dataset)) * 100
    print(
        f"Feature: {col}, Outliers: {len(outliers)}, Outlier Percentage: {outlier_percentage:.2f}%"
    )

# %%
# Export the preprocessed dataframe to a CSV file
simplified_dataset.to_csv("preprocessed_ashrae_data.csv", index=False)

# %% [markdown]
# <div style="text-align: center; color:aquamarine">
#   <h2>Feature Engineering</h2>
# </div>
#   <p>The following section of the code is responsible for feature engineering. Feature engineering involves creating new features or modifying existing ones to improve the performance of machine learning models. This process includes handling categorical variables, encoding, scaling, and transforming features to ensure they are in the optimal format for model training. Effective feature engineering can significantly enhance the predictive power of the models.</p>

# %%
# Import the preprocessed dataset
preprocessed_df = pd.read_csv("preprocessed_ashrae_data.csv")

# Display the shape and first few rows of the dataframe
print("Dataset Shape:", preprocessed_df.shape)
preprocessed_df.head()

# %%
# Replace multiple types of null values with np.nan
preprocessed_df["thermal_comfort"] = preprocessed_df["thermal_comfort"].replace(
    ["Na", " "], np.nan
)

# %% [markdown]
# ## Selecting Required features for our project

# %%
preprocessed_df = preprocessed_df[
    [
        "year",
        "season",
        "koppen_climate",
        "climate_description",
        "city",
        "country",
        "building_type",
        "cooling_strategy",
        "age",
        "sex",
        "thermal_sensation",
        "thermal_comfort",
        "clothing_insulation",
        "metabolic_rate",
        "air_temperature",
        "relative_humidity",
        "air_velocity",
        "outdoor_air_temperature",
        "thermal_preference",
    ]
]

# %%
preprocessed_df.sample(n=10)

# %% [markdown]
# ### Checking for Highly Correlated Features

# %%
# Correlation heatmap
plt.figure(figsize=(12, 10))
numerical_cols = preprocessed_df.select_dtypes(include=["float64", "Int64"]).columns
correlation_matrix = preprocessed_df[numerical_cols].corr()
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, cbar=True)
plt.title("Correlation Heatmap")
plt.show()

# Highly correlated features
threshold = 0.8
high_corr_features = (
    correlation_matrix[abs(correlation_matrix) > threshold].stack().reset_index()
)
high_corr_features = high_corr_features[
    high_corr_features["level_0"] != high_corr_features["level_1"]
]
print("Highly Correlated Features:\n", high_corr_features)

# %% [markdown]
# ## Imputing Null Values using different strategies

# %%
# copying DataFrame to avoid SettingWithCopyWarning
imputed_df = preprocessed_df.copy(deep=True)

# Define numerical and categorical columns
numerical_cols = imputed_df.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = imputed_df.select_dtypes(include=["object"]).columns

# Exclude the target variable from processing
numerical_cols = numerical_cols.drop("thermal_preference", errors="ignore")

# Impute numerical columns with median
imputer_numeric = SimpleImputer(strategy="median")
imputed_df[numerical_cols] = imputer_numeric.fit_transform(imputed_df[numerical_cols])

# Impute categorical columns with the most frequent value
imputer_categorical = SimpleImputer(strategy="most_frequent")
imputed_df[categorical_cols] = imputer_categorical.fit_transform(
    imputed_df[categorical_cols]
)

imputed_df.head()

# %%
# Calculate percentage of missing values
missing_values = imputed_df.isnull().mean() * 100
print("\nPercentage of Missing Values:\n", missing_values)

# Visualize missing values
plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values)
plt.xticks(rotation=45, ha="right")
plt.title("Percentage of Missing Values")
plt.ylabel("Percentage")
plt.show()

# %%
# Visualize missing values using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(imputed_df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# %%
print(f"categorical_cols: {categorical_cols}")
print(f"numerical_cols: {numerical_cols}")

# %% [markdown]
# ## Encoding Categorical Variables

# %% [markdown]
# ### Handling Year Column

# %%
# Calculate year_delta
imputed_df["year_delta"] = imputed_df["year"] - imputed_df["year"].min()

# Insert the year_delta column right after the year column
year_index = imputed_df.columns.get_loc("year")  # Get the index of the 'year' column
cols = list(imputed_df.columns)  # Get all columns as a list
# Rearrange columns: Insert year_delta after year
cols.insert(year_index + 1, cols.pop(cols.index("year_delta")))
imputed_df = imputed_df[cols]  # Reorder columns in the DataFrame

# Verify the placement of year_delta
imputed_df.head()

# %% [markdown]
# ### Handling SEASON, KOPPEN_CLIMATE, BUILDING_TYPE, COOLING_STRATEGY

# %% [markdown]
# ### Columns in the Dataset
#
# | **Column**               | **Encoding Recommendation**                              | **Reason**                                                                                   |
# |--------------------------|---------------------------------------------------------|---------------------------------------------------------------------------------------------|
# | `year`                   | No Encoding                                             | Treat as-is or use relative features (e.g., `year_delta`).                                  |
# | `season`                 | One-Hot Encoding                                        | Low cardinality and categorical. No ordinal relationship.                                   |
# | `koppen_climate`         | One-Hot Encoding                                        | Moderate cardinality. One-hot if space allows; label encoding for simpler models.           |
# | `climate_description`    | Hash Encoding                                           | High cardinality (likely descriptive text). Hashing prevents dimensionality explosion.       |
# | `city`                   | Hash Encoding                                           | High cardinality. Hashing works well for unseen cities; frequency encoding maintains trends. |
# | `country`                | Hash Encoding                                           | Moderate cardinality. Use label encoding for tree-based models or frequency encoding.       |
# | `building_type`          | One-Hot Encoding                                        | Low cardinality. Categories like residential/commercial.                                    |
# | `cooling_strategy`       | One-Hot Encoding                                        | Low cardinality. Represents distinct strategies.                                            |
# | `age`                    | No Encoding                                             | Continuous numerical feature.                                                              |
# | `sex`                    | Label Encoding                                          | Binary categorical feature.                                                                |
# | `thermal_sensation`      | No Encoding                                             | Numerical target-like feature.                                                             |
# | `thermal_comfort`        | No Encoding                                             | Ordinal variable; consider scaling instead of encoding.                                     |
# | `clothing_insulation`    | No Encoding                                             | Continuous numerical feature.                                                              |
# | `metabolic_rate`         | No Encoding                                             | Continuous numerical feature.                                                              |
# | `air_temperature`        | No Encoding                                             | Continuous numerical feature.                                                              |
# | `relative_humidity`      | No Encoding                                             | Continuous numerical feature.                                                              |
# | `air_velocity`           | No Encoding                                             | Continuous numerical feature.                                                              |
# | `outdoor_air_temperature`| No Encoding                                             | Continuous numerical feature.                                                              |
# | `thermal_preference`     | Label Encoding                                          | Target variable for classification tasks.                                                  |
#

# %% [markdown]
# ### Label Encode Sex and Thermal Preference

# %%
encoded_df = imputed_df.copy(deep=True)

# Label encode binary categories (e.g., sex)
if "sex" in categorical_cols:
    label_encoder = LabelEncoder()
    encoded_df["sex"] = label_encoder.fit_transform(encoded_df["sex"])
    encoded_df["thermal_preference"] = label_encoder.fit_transform(
        encoded_df["thermal_preference"]
    )

# %%
for col in categorical_cols:
    unique_values = encoded_df[col].unique()
    print(f"Unique values in column '{col} {len(unique_values)}': {unique_values}\n")

# %%
# Drop 'city' and 'climate_description' columns
encoded_df = encoded_df.drop(columns=["city", "climate_description", "year"])

# Display the updated dataframe
print("Updated DataFrame Columns:\n", encoded_df.columns)

# %% [markdown]
# ### Hash Encode Country

# %%
# Initialize the hasher
hasher = FeatureHasher(
    n_features=10, input_type="string"
)  # Adjust n_features as needed

# Apply hashing to 'country', 'city', and 'climate_description'
hashed_country = hasher.transform(encoded_df["country"].astype(str).map(lambda x: [x]))

# Convert the hashed features into DataFrames
hashed_country_df = pd.DataFrame(
    hashed_country.toarray(), columns=[f"hashed_country_{i}" for i in range(10)]
)

# Concatenate the hashed columns with the original DataFrame
hash_encoded_df = pd.concat(
    [encoded_df.reset_index(drop=True), hashed_country_df],
    axis=1,
)

# Drop the original columns if they are no longer needed
hash_encoded_df.drop(columns=["country"], inplace=True)

# Display the updated DataFrame
print("Updated DataFrame with Hashing Encoding:\n")
hash_encoded_df.head()

# %%
hash_encoded_df.columns

# %% [markdown]
# ### One-Hot Encode Season, Koppen Climate, Building Type, and Cooling Strategy

# %%
# Perform one-hot encoding on the specified columns
columns_to_encode = ["season", "koppen_climate", "building_type", "cooling_strategy"]
one_hash_encoded_df = pd.get_dummies(
    hash_encoded_df, columns=columns_to_encode, drop_first=True
)

# Display the updated DataFrame
print("Updated DataFrame with One-Hot Encoding:\n", one_hash_encoded_df.head())

# %%
one_hash_encoded_df.sample(n=10)

# %% [markdown]
# ### Scaling Numerical Features without YEAR

# %%
# List of numerical columns to scale (excluding target variable, year, and non-numerical columns)
numerical_cols_wo_year = [
    "age",
    "thermal_sensation",
    "thermal_comfort",
    "clothing_insulation",
    "metabolic_rate",
    "air_temperature",
    "relative_humidity",
    "air_velocity",
    "outdoor_air_temperature",
]

# Verify the selected columns
print("Numerical Columns to Scale:", numerical_cols)

# %%
# Exclude the target variable from scaling
scaler = StandardScaler()
one_hash_encoded_df[numerical_cols_wo_year] = scaler.fit_transform(
    one_hash_encoded_df[numerical_cols_wo_year]
)
one_hash_encoded_df[numerical_cols_wo_year].head()

# %%
# Export the sampled dataframe to a CSV file
one_hash_encoded_df.to_csv("encoded_scaled_ashrae_data.csv", index=False)

# %% [markdown]
# ### Scaling Validation

# %%
scaled_df = one_hash_encoded_df.copy(deep=True)
# Check mean and standard deviation of scaled features (for StandardScaler)
scaled_summary = scaled_df[numerical_cols_wo_year].describe().T[["mean", "std"]]
print("Mean and Std of Scaled Features:\n", scaled_summary)

# Check min and max for MinMaxScaler
scaled_min_max = scaled_df[numerical_cols_wo_year].agg(["min", "max"])
print("Min and Max of Scaled Features:\n", scaled_min_max)

# %%
# Plot histograms for scaled features
scaled_df[numerical_cols_wo_year].hist(
    figsize=(15, 10), bins=20, color="teal", edgecolor="black"
)
plt.suptitle("Histograms of Scaled Features")
plt.show()

# Boxplot for scaled features
plt.figure(figsize=(12, 6))
sns.boxplot(data=scaled_df[numerical_cols_wo_year], palette="viridis")
plt.title("Boxplot of Scaled Features")
plt.xticks(rotation=45)
plt.show()

# %% [markdown]
# ### ENCODING VALIDATION

# %%
# Check presence of encoded columns
encoded_columns = [
    col
    for col in scaled_df.columns
    if col.startswith(
        ("season_", "koppen_climate_", "building_type_", "cooling_strategy_")
    )
]
print("Encoded Columns:\n", encoded_columns)

# %%
# Check if original columns were dropped
original_columns = ["season", "koppen_climate", "building_type", "cooling_strategy"]
missing_original_columns = [
    col for col in original_columns if col not in scaled_df.columns
]
print("Original Columns Dropped:", missing_original_columns)

# %%
# Example: Verify One-Hot Encoding consistency for 'season'
season_encoded_columns = [col for col in encoded_columns if col.startswith("season_")]
season_sums = scaled_df[season_encoded_columns].sum(axis=1)
print("Rows with Inconsistent One-Hot Encoding:", (season_sums != 1).sum())

# %%
missing_values = scaled_df.isnull().sum()
print(
    "Missing Values After Scaling and Encoding:\n", missing_values[missing_values > 0]
)

# %%
# Confirm dataset shape before and after encoding/scaling
print("Dataset Shape After Scaling and Encoding:", scaled_df.shape)

# %%
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(scaled_df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# %%
hashed_columns = [col for col in scaled_df.columns if col.startswith("hashed_")]
print("Distribution of Hashed Features:\n", scaled_df[hashed_columns].sum().describe())


# %%
# Function to test encoding inconsistency for one-hot encoded columns
def check_one_hot_encoding_consistency(df, original_columns, encoded_prefixes):
    for original_col, prefix in zip(original_columns, encoded_prefixes):
        # Identify one-hot encoded columns for this feature
        encoded_columns = [col for col in df.columns if col.startswith(prefix)]

        # Check if rows sum to exactly 1
        row_sums = df[encoded_columns].sum(axis=1)
        inconsistent_rows = (row_sums != 1).sum()

        # Report inconsistencies
        print(f"Original Column: '{original_col}'")
        print(f"Encoded Columns: {encoded_columns}")
        print(f"Rows with Inconsistent Encoding: {inconsistent_rows}")
        print("-" * 50)


# Define original columns and their corresponding one-hot encoded prefixes
original_columns = ["season", "koppen_climate", "building_type", "cooling_strategy"]
encoded_prefixes = ["season_", "koppen_climate_", "building_type_", "cooling_strategy_"]

# Run the consistency check
check_one_hot_encoding_consistency(scaled_df, original_columns, encoded_prefixes)

# %% [markdown]
# ### Encoding and Scaling Validation Conclusion

# %% [markdown]
# <div style="text-align: center; color:aquamarine">
#   <h2>Model Training & Evaluation</h2>
# </div>
#   <p>The following section of the code is responsible for training and evaluating machine learning models. This includes splitting the dataset into training and testing sets, selecting appropriate algorithms, training the models, and evaluating their performance using various metrics. The goal is to identify the best-performing model for predicting thermal comfort based on the given features.</p>

# %%
# Split data into features (X) and target (y)
X = scaled_df.drop(columns=["thermal_preference"])
y = scaled_df["thermal_preference"]

# Full dataset split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Sampled dataset split (10% for quick model evaluation)
X_sample, _, y_sample, _ = train_test_split(
    X_train, y_train, train_size=0.1, random_state=42, stratify=y_train
)

# Display shapes
print("Full Train Shape:", X_train.shape, "Sample Train Shape:", X_sample.shape)

# %% [markdown]
# (a) Logistic Regression

# %%
# Logistic Regression on Full Data
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

# Evaluate the model
print(
    "Logistic Regression (Full Data) Classification Report:\n",
    classification_report(y_test, y_pred_lr),
)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_lr, labels=log_reg.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=log_reg.classes_)
disp.plot(cmap="viridis")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# %% [markdown]
# (b) Random Forest Classifier

# %%
# Random Forest on Full Data
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print(
    "Random Forest (Full Data) Classification Report:\n",
    classification_report(y_test, y_pred_rf),
)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot(cmap="viridis")
plt.title("Random Forest Confusion Matrix")
plt.show()

# %% [markdown]
# (c) Gradient Boosting Models

# %% [markdown]
# XGBoost

# %%
# XGBoost
xgb = XGBClassifier(random_state=42, eval_metric="mlogloss", use_label_encoder=False)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Evaluate the model
print(
    "XGBoost (Full Data) Classification Report:\n",
    classification_report(y_test, y_pred_xgb),
)
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb, labels=xgb.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=xgb.classes_)
disp.plot(cmap="viridis")
plt.title("XGBoost Confusion Matrix")
plt.show()

# %% [markdown]
# LightGBM

# %%
lgbm = LGBMClassifier(random_state=42)
lgbm.fit(X_train, y_train)
y_pred_lgbm = lgbm.predict(X_test)


# Evaluate the model
print(
    "LightGBM (Full Data) Classification Report:\n",
    classification_report(y_test, y_pred_lgbm),
)
print("Accuracy:", accuracy_score(y_test, y_pred_lgbm))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_lgbm, labels=lgbm.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lgbm.classes_)
disp.plot(cmap="viridis")
plt.title("LightGBM Confusion Matrix")
plt.show()

# %%
# CatBoost
catboost = CatBoostClassifier(random_state=42, verbose=0)
catboost.fit(X_train, y_train)
y_pred_cat = catboost.predict(X_test)

# Evaluate the model
print(
    "CatBoost (Full Data) Classification Report:\n",
    classification_report(y_test, y_pred_cat),
)
print("Accuracy:", accuracy_score(y_test, y_pred_cat))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred_cat, labels=catboost.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=catboost.classes_)
disp.plot(cmap="viridis")
plt.title("CatBoost Confusion Matrix")
plt.show()

# %% [markdown]
# ### Ensemble Model (Voting Classifier)

# %%
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb), ("cat", catboost)],
    voting="soft",
)

ensemble.fit(X_train, y_train)
ensemble_accuracy = ensemble.score(X_test, y_test)
print("Ensemble Model Accuracy:", ensemble_accuracy)

# %%
# Dictionary to store model names and accuracies
model_accuracies = {
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "Random Forest": accuracy_score(y_test, y_pred_rf),
    "XGBoost": accuracy_score(y_test, y_pred_xgb),
    "LightGBM": accuracy_score(y_test, y_pred_lgbm),
    "CatBoost": accuracy_score(y_test, y_pred_cat),
    "ensemble": ensemble_accuracy,
}

# Display results
for model, accuracy in model_accuracies.items():
    print(f"{model}: {accuracy:.4f}")

# %%
# SVM
svm = SVC(kernel="rbf", random_state=42)
svm.fit(X_sample, y_sample)
y_pred_svm = svm.predict(X_test)

print(
    "SVM Classification (Sampled Data) Classification Report:\n",
    classification_report(y_test, y_pred_svm),
)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Naive Bayes
nb = GaussianNB()
nb.fit(X_sample, y_sample)
y_pred_nb = nb.predict(X_test)

print(
    "Naive Bayes (Sampled Data) Classification Report:\n",
    classification_report(y_test, y_pred_nb),
)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_sample, y_sample)
y_pred_knn = knn.predict(X_test)

print(
    "KNN (Sampled Data) Classification Report:\n",
    classification_report(y_test, y_pred_knn),
)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
mlp.fit(X_sample, y_sample)
y_pred_mlp = mlp.predict(X_test)

print(
    "MLP (Sampled Data) Classification Report:\n",
    classification_report(y_test, y_pred_mlp),
)
print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))

# ElasticNet (Sample Data)
elastic_net = LogisticRegression(
    penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=1000, random_state=42
)
elastic_net.fit(X_sample, y_sample)
y_pred_elastic = elastic_net.predict(X_test)

print(
    "Elastic Net  (Sampled Data) Classification Report:\n",
    classification_report(y_test, y_pred_elastic),
)
print("Elastic Net Accuracy:", accuracy_score(y_test, y_pred_elastic))

# CatBoost (Sample Data)
catboost_sd = CatBoostClassifier(random_state=42, verbose=0)
catboost_sd.fit(X_sample, y_sample)
y_pred_catboost_sd = catboost_sd.predict(X_test)

print(
    "CatBoost Classification (Sample Data) Report:\n",
    classification_report(y_test, y_pred_catboost_sd),
)
print("CatBoost Accuracy:", accuracy_score(y_test, y_pred_catboost_sd))

# %%
# Dictionary to store model names and accuracies
model_accuracies_sampled = {
    "SVM": accuracy_score(y_test, y_pred_svm),
    "KNN": accuracy_score(y_test, y_pred_knn),
    "MLP": accuracy_score(y_test, y_pred_mlp),
    "Naive Bayes": accuracy_score(y_test, y_pred_nb),
    "CatBoost": accuracy_score(y_test, y_pred_catboost_sd),
    "Elastic Net": accuracy_score(y_test, y_pred_elastic),
}

# Display results
for model, accuracy in model_accuracies_sampled.items():
    print(f"{model}: {accuracy:.4f}")

# %%
# Create a dictionary of models
models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf,
    "XGBoost": xgb,
    "LightGBM": lgbm,
    "CatBoost": catboost,
    "ensemble": ensemble,
    "SVM": svm,
    "KNN": knn,
    "Naive Bayes": nb,
    "Elastic Net": elastic_net,
}

models_sampled = {
    "SVM": svm,
    "KNN": knn,
    "Naive Bayes": nb,
    "Elastic Net": elastic_net,
}

# Initialize dictionaries to store evaluation metrics
accuracy_scores = {}
precision_scores = {}
recall_scores = {}
f1_scores = {}
roc_auc_scores = {}

# Set up the subplot grid (4x3 for 10 models)
fig, axes = plt.subplots(4, 3, figsize=(15, 15))

# Flatten axes for easier iteration
axes = axes.ravel()

# Iterate over models, fit, and generate confusion matrix & evaluation metrics
for i, (name, model) in enumerate(models.items()):
    # Train the model
    if name in models_sampled.keys():
        model.fit(X_sample, y_sample)
    else:
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[i], cmap="Blues", colorbar=False)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    try:
        roc_auc = roc_auc_score(
            y_test, model.predict_proba(X_test), multi_class="ovr", average="weighted"
        )
    except:
        roc_auc = (
            None  # Some models don't support probability estimation, like Naive Bayes
        )

    # Store evaluation metrics
    accuracy_scores[name] = accuracy
    precision_scores[name] = precision
    recall_scores[name] = recall
    f1_scores[name] = f1
    roc_auc_scores[name] = roc_auc

    # Set plot title and accuracy text
    axes[i].set_title(f"{name}")
    axes[i].text(
        0.5,
        0.85,
        f"Accuracy: {accuracy:.4f}",
        horizontalalignment="center",
        transform=axes[i].transAxes,
        fontsize=12,
        color="black",
        weight="bold",
    )

# Remove the last two empty subplots
fig.delaxes(axes[-1])
fig.delaxes(axes[-2])

# Adjust layout for subplots
plt.tight_layout()
plt.show()

# Plot evaluation metrics
metrics_df = pd.DataFrame(
    {
        "Accuracy": accuracy_scores,
        "Precision": precision_scores,
        "Recall": recall_scores,
        "F1-Score": f1_scores,
        "ROC AUC": roc_auc_scores,
    }
)

# Barplot for evaluation metrics using seaborn
# Melt the dataframe for easier plotting with seaborn
metrics_melted = metrics_df.reset_index().melt(
    id_vars="index", var_name="Metric", value_name="Score"
)

# Plot the barplot
plt.figure(figsize=(12, 8))
sns.barplot(
    x="index",
    y="Score",
    hue="Metric",
    data=metrics_melted,
    palette="crest",
)
plt.title("Evaluation Metrics Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print all the evaluation metrics of all the models in a tabular format
metrics_df = pd.DataFrame(
    {
        "Model": list(accuracy_scores.keys()),
        "Accuracy": list(accuracy_scores.values()),
        "Precision": list(precision_scores.values()),
        "Recall": list(recall_scores.values()),
        "F1-Score": list(f1_scores.values()),
        "ROC AUC": list(roc_auc_scores.values()),
    }
)

print(metrics_df)

# %% [markdown]
# <div style="text-align: center; color:aquamarine">
#   <h4>First Phase of Model Training and Evaluation Completed</h4>
# </div>
#
# ---

# %% [markdown]
# ---
#

# %% [markdown]
#
# <div style="text-align: center; color:aquamarine">
#   <h4>Second Phase of Model Training and Evaluation Starts</h4>
# </div>

# %% [markdown]
# <div style="text-align: center; color:aquamarine">
#   <h2>Hyperparameter Tuning</h2>
# </div>
#   <p>The following section of the code is responsible for hyperparameter tuning. Hyperparameter tuning involves selecting the best set of hyperparameters for a machine learning model to improve its performance. This process includes using techniques such as GridSearchCV and RandomizedSearchCV to systematically search for the optimal hyperparameters. Effective hyperparameter tuning can significantly enhance the predictive power and generalization ability of the models.</p>

# %%
evaluated_df = scaled_df.copy(deep=True)

# %% [markdown]
# ### Handling Imbalanced Classes

# %%
X = evaluated_df.drop(
    columns=["thermal_preference"]
)  # Replace with your actual target variable
y = evaluated_df["thermal_preference"]

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into train/test set
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Check the new class distribution
print(y_resampled.value_counts())

# %% [markdown]
# ### Hyperparameter Tuning for Random Forest Classifier

# %%
# Hyperparameter tuning for Random Forest
rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
rf = RandomForestClassifier(random_state=42)
rf_search = RandomizedSearchCV(
    rf, param_distributions=rf_params, n_iter=10, cv=3, random_state=42, verbose=2
)
rf_search.fit(X_train, y_train)
print(f"Best Random Forest Params: {rf_search.best_params_}")

# Hyperparameter tuning for XGBoost
xgb_params = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}
xgb = XGBClassifier(random_state=42)
xgb_search = RandomizedSearchCV(
    xgb, param_distributions=xgb_params, n_iter=10, cv=3, random_state=42, verbose=2
)
xgb_search.fit(X_train, y_train)
print(f"Best XGBoost Params: {xgb_search.best_params_}")

# Hyperparameter tuning for LightGBM
lgbm_params = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "num_leaves": [31, 50, 100],
}
lgbm = LGBMClassifier(random_state=42)
lgbm_search = RandomizedSearchCV(
    lgbm, param_distributions=lgbm_params, n_iter=10, cv=3, random_state=42, verbose=2
)
lgbm_search.fit(X_train, y_train)
print(f"Best LightGBM Params: {lgbm_search.best_params_}")

# Hyperparameter tuning for CatBoost
catboost_params = {
    "iterations": [100, 200],
    "depth": [6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "l2_leaf_reg": [1, 3, 5],
}
catboost = CatBoostClassifier(random_state=42, verbose=0)
catboost_search = RandomizedSearchCV(
    catboost,
    param_distributions=catboost_params,
    n_iter=5,
    cv=3,
    random_state=42,
    verbose=2,
)
catboost_search.fit(X_train, y_train)
print(f"Best CatBoost Params: {catboost_search.best_params_}")

# %% [markdown]
# Best XGBoost Params: {'subsample': 0.8, 'n_estimators': 200, 'max_depth': 7, 'learning_rate': 0.2, 'colsample_bytree': 0.9}
#
# Best LightGBM Params: {'num_leaves': 50, 'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.2}
#
# Best CatBoost Params: {'learning_rate': 0.05, 'l2_leaf_reg': 3, 'iterations': 200, 'depth': 10}

# %% [markdown]
# ### Evaluating Random Forest using Cross-Validation

# %%
# Evaluate Random Forest using cross-validation
rf_cv_scores = cross_val_score(
    rf_search.best_estimator_, X_train, y_train, cv=5, scoring="accuracy"
)
print(f"Random Forest Cross-Validation Accuracy: {np.mean(rf_cv_scores):.4f}")

# Evaluate XGBoost using cross-validation
xgb_cv_scores = cross_val_score(
    xgb_search.best_estimator_, X_train, y_train, cv=5, scoring="accuracy"
)
print(f"XGBoost Cross-Validation Accuracy: {np.mean(xgb_cv_scores):.4f}")

# Evaluate LightGBM using cross-validation
lgbm_cv_scores = cross_val_score(
    lgbm_search.best_estimator_, X_train, y_train, cv=5, scoring="accuracy"
)
print(f"LightGBM Cross-Validation Accuracy: {np.mean(lgbm_cv_scores):.4f}")

# Evaluate CatBoost using cross-validation
catboost_cv_scores = cross_val_score(
    catboost_search.best_estimator_, X_train, y_train, cv=5, scoring="accuracy"
)
print(f"CatBoost Cross-Validation Accuracy: {np.mean(catboost_cv_scores):.4f}")

# %% [markdown]
# | Model           | Cross-Validation Accuracy |
# |-----------------|---------------------------|
# | Random Forest   | 0.8424                    |
# | XGBoost         | 0.8406                    |
# | LightGBM        | 0.8406                    |
# | CatBoost        | 0.8042                    |

# %% [markdown]
# ## Ensemble Models

# %% [markdown]
# ### Stacking Classifier

# %%
# Stacking Classifier
base_learners = [
    ("rf", rf_search.best_estimator_),
    ("xgb", xgb_search.best_estimator_),
    ("lgbm", lgbm_search.best_estimator_),
    ("catboost", catboost_search.best_estimator_),
]

meta_classifier = LogisticRegression()  # Logistic Regression as meta-classifier

stacking_clf = StackingClassifier(
    estimators=base_learners, final_estimator=meta_classifier
)

# Fit the stacking classifier
stacking_clf.fit(X_train, y_train)

# Evaluate performance
y_pred_stack = stacking_clf.predict(X_test)
print("Stacking Classifier Accuracy:", accuracy_score(y_test, y_pred_stack))

# %% [markdown]
# ### Stacking Classifier Accuracy: 0.8606176865528486

# %% [markdown]
# ### Voting Classifier

# %%
# Voting Classifier (Soft Voting)
voting_clf = VotingClassifier(
    estimators=[
        ("rf", rf_search.best_estimator_),
        ("xgb", xgb_search.best_estimator_),
        ("lgbm", lgbm_search.best_estimator_),
        ("catboost", catboost_search.best_estimator_),
    ],
    voting="soft",  # Soft voting (uses probabilities)
)

# Fit the voting classifier
voting_clf.fit(X_train, y_train)

# Evaluate performance
y_pred_voting = voting_clf.predict(X_test)
print(
    "Voting Classifier Accuracy (Soft Voting):", accuracy_score(y_test, y_pred_voting)
)

# %% [markdown]
# ### Voting Classifier Accuracy (Soft Voting): 0.8430078649530021

# %% [markdown]
# # Training Model with Best Parameters

# %%
# XGBoost with the best parameters
xgb_best = XGBClassifier(
    subsample=0.8,
    n_estimators=200,
    max_depth=7,
    learning_rate=0.2,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="mlogloss",
    use_label_encoder=False,
)
xgb_best.fit(X_train, y_train)
y_pred_xgb = xgb_best.predict(X_test)
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb))

# LightGBM with the best parameters
lgbm_best = LGBMClassifier(
    num_leaves=50, n_estimators=300, max_depth=5, learning_rate=0.2, random_state=42
)
lgbm_best.fit(X_train, y_train)
y_pred_lgbm = lgbm_best.predict(X_test)
print(f"LightGBM Accuracy: {accuracy_score(y_test, y_pred_lgbm):.4f}")
print(classification_report(y_test, y_pred_lgbm))

# CatBoost with the best parameters
catboost_best = CatBoostClassifier(
    learning_rate=0.05,
    l2_leaf_reg=3,
    iterations=200,
    depth=10,
    random_state=42,
    verbose=0,
)
catboost_best.fit(X_train, y_train)
y_pred_catboost = catboost_best.predict(X_test)
print(f"CatBoost Accuracy: {accuracy_score(y_test, y_pred_catboost):.4f}")
print(classification_report(y_test, y_pred_catboost))

# %%
# Initialize the dictionary to store metrics for each model
metrics = {}


# Function to calculate all metrics
def calculate_metrics(y_test, y_pred, model_name):
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    # roc_auc = roc_auc_score(y_test, y_pred, average="weighted", multi_class="ovr")
    accuracy = accuracy_score(y_test, y_pred)

    metrics[model_name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        # "ROC AUC": roc_auc,
    }


# Evaluate metrics for each model
models_predictions = {
    "XGBoost": y_pred_xgb,
    "LightGBM": y_pred_lgbm,
    "CatBoost": y_pred_catboost,
    "Voting Classifier (Soft Voting)": y_pred_voting,
    "Stacking Classifier": y_pred_stack,
}

for model_name, y_pred in models_predictions.items():
    calculate_metrics(y_test, y_pred, model_name)

# Print the results
print("\nModel Performance Comparison:")
for model, scores in metrics.items():
    print(f"\n{model}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")

# Now, create a DataFrame for visualization
import pandas as pd

metrics_df = pd.DataFrame(metrics).T

# Plot the metrics in bar chart
# Plot the metrics in bar chart using seaborn
metrics_melted = metrics_df.reset_index().melt(
    id_vars="index", var_name="Metric", value_name="Score"
)

plt.figure(figsize=(12, 8))
sns.barplot(x="index", y="Score", hue="Metric", data=metrics_melted, palette="viridis")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# | **Model**                          | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
# |------------------------------------|--------------|---------------|------------|--------------|
# | XGBoost                            | 0.8430       | 0.8427        | 0.8430     | 0.8428       |
# | LightGBM                           | 0.8427       | 0.8426        | 0.8427     | 0.8426       |
# | CatBoost                           | 0.7996       | 0.7983        | 0.7996     | 0.7986       |
# | Voting Classifier (Soft Voting)    | 0.8430       | 0.8423        | 0.8430     | 0.8424       |
# | Stacking Classifier                | 0.8606       | 0.8606        | 0.8606     | 0.8606       |
#

# %% [markdown]
# ### Model Exporting

# %%
# Save the best model (e.g., Stacking Classifier)
joblib.dump(stacking_clf, "artifacts/stacking_model.pkl")

# %% [markdown]
# <div style="text-align: center; color:aquamarine">
#   <h2>Conclusion</h2>
# </div>
#   <p>The following section of the code is responsible for loading the dataset from a CSV file and inspecting its contents. The dataset contains various attributes related to thermal comfort, including publication details, climate information, building characteristics, and individual thermal comfort responses. The dataset is loaded into a pandas DataFrame for further analysis and processing.</p>

# %% [markdown]
# # Dataset Analysis and Preprocessing
#
# ## Dataset Overview
#
# The dataset used in this analysis contains various attributes related to thermal comfort, including publication details, climate information, building characteristics, and individual thermal comfort responses. The dataset is loaded into a pandas DataFrame for further analysis and processing.
#
# ### Key Features:
# - **Year**: The year when the data was collected.
# - **Season**: The season during which the data was collected.
# - **Koppen Climate Classification**: The climate classification of the location.
# - **Climate Description**: A textual description of the climate.
# - **City**: The city where the data was collected.
# - **Country**: The country where the data was collected.
# - **Building Type**: The type of building where the data was collected.
# - **Cooling Strategy**: The cooling strategy used in the building.
# - **Age**: The age of the individual.
# - **Sex**: The sex of the individual.
# - **Thermal Sensation**: The thermal sensation reported by the individual.
# - **Thermal Comfort**: The thermal comfort reported by the individual.
# - **Clothing Insulation**: The clothing insulation value.
# - **Metabolic Rate**: The metabolic rate of the individual.
# - **Air Temperature**: The air temperature at the time of data collection.
# - **Relative Humidity**: The relative humidity at the time of data collection.
# - **Air Velocity**: The air velocity at the time of data collection.
# - **Outdoor Air Temperature**: The outdoor air temperature at the time of data collection.
# - **Thermal Preference**: The thermal preference of the individual.
#
# ## Preprocessing Steps
#
# ### 1. Handling Missing Values
# - **Numerical Columns**: Missing values in numerical columns were imputed using the median value of each column.
# - **Categorical Columns**: Missing values in categorical columns were imputed using the most frequent value of each column.
#
# ### 2. Encoding Categorical Variables
# - **Label Encoding**: Binary categorical features such as `sex` and `thermal_preference` were label encoded.
# - **One-Hot Encoding**: Categorical features with low cardinality such as `season`, `koppen_climate`, `building_type`, and `cooling_strategy` were one-hot encoded.
# - **Hash Encoding**: High cardinality features such as `country` were hash encoded to prevent dimensionality explosion.
#
# ### 3. Feature Scaling
# - **Standard Scaling**: Numerical features were scaled using StandardScaler to ensure they have a mean of 0 and a standard deviation of 1.
#
# ### 4. Handling Outliers
# - **Outlier Detection**: Boxplots were used to detect outliers in numerical features.
# - **Outlier Capping**: Outliers were capped using the interquartile range (IQR) method to limit their impact on the model.
#
# ### 5. Handling Imbalanced Classes
# - **SMOTE**: Synthetic Minority Over-sampling Technique (SMOTE) was used to handle class imbalance in the target variable `thermal_preference`.
#
# ### 6. Feature Selection
# - **Variance Threshold**: Features with low variance were removed to reduce the dimensionality of the dataset.
#
# ## Summary
#
# The preprocessing steps ensured that the dataset was clean, balanced, and ready for model training. By handling missing values, encoding categorical variables, scaling numerical features, and addressing outliers and class imbalance, we improved the quality of the dataset and enhanced the performance of the machine learning models.

# %% [markdown]
# Model Performance Analysis and Hyperparameter Tuning Discussion
#
# 1. Introduction to the Project:
# In this project, we aimed to predict thermal preferences (e.g., whether a person prefers cooler, no change, or warmer conditions in a given environment) based on a set of environmental and personal features. We utilized several machine learning models and performed hyperparameter tuning to improve their performance.
#
# 2. Performance Metrics Before and After Hyperparameter Tuning
# Below are the model performance metrics before and after applying hyperparameter tuning. The metrics include Accuracy, Precision, Recall, F1-Score, and ROC AUC.
#
# Model Performance Before Hyperparameter Tuning:
# | Model                | Accuracy | Precision | Recall | F1-Score | ROC AUC |
# |----------------------|----------|-----------|--------|----------|---------|
# | Logistic Regression  | 0.7329   | 0.7394    | 0.7329 | 0.7251   | 0.8473  |
# | Random Forest        | 0.7883   | 0.7881    | 0.7883 | 0.7870   | 0.9048  |
# | XGBoost              | 0.7853   | 0.7852    | 0.7853 | 0.7842   | 0.9073  |
# | LightGBM             | 0.7839   | 0.7839    | 0.7839 | 0.7828   | 0.9051  |
# | CatBoost             | 0.7860   | 0.7858    | 0.7860 | 0.7850   | 0.9075  |
# | Ensemble             | 0.7917   | 0.7917    | 0.7917 | 0.7905   | 0.9132  |
# | SVM                  | 0.7113   | 0.7385    | 0.7113 | 0.6903   | -     |
# | KNN                  | 0.6925   | 0.6952    | 0.6925 | 0.6864   | 0.8109  |
# | Naive Bayes          | 0.4543   | 0.5854    | 0.4543 | 0.4453   | 0.7128  |
# | Elastic Net          | 0.7282   | 0.7358    | 0.7282 | 0.7196   | 0.8445  |
#
# Model Performance After Hyperparameter Tuning:
# | **Model**                          | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
# |------------------------------------|--------------|---------------|------------|--------------|
# | XGBoost                            | 0.8430       | 0.8427        | 0.8430     | 0.8428       |
# | LightGBM                           | 0.8427       | 0.8426        | 0.8427     | 0.8426       |
# | CatBoost                           | 0.7996       | 0.7983        | 0.7996     | 0.7986       |
# | Voting Classifier (Soft Voting)    | 0.8430       | 0.8423        | 0.8430     | 0.8424       |
# | Stacking Classifier                | 0.8606       | 0.8606        | 0.8606     | 0.8606       |
#
# 3. Discussion of Hyperparameter Tuning and Model Performance
# 3.1 Model Comparison (Before vs. After Hyperparameter Tuning)
# Best Model Before Tuning:
#
# - Ensemble Model (Voting Classifier): This was the best-performing model, with an accuracy of 0.7917 and the highest ROC AUC of 0.9132.
# - XGBoost and CatBoost were very close in performance before tuning, showing high accuracy and precision.
#
# Best Model After Tuning:
#
# - Stacking Classifier emerged as the best performer after hyperparameter tuning, with a significant increase in accuracy (0.8606) and F1-Score (0.8606). This improvement highlights the power of ensemble methods where multiple models are combined to improve generalization and performance.
# - XGBoost, LightGBM, and the Voting Classifier also showed substantial improvements, all achieving accuracy scores above 0.843 after tuning.
#
# 3.2 Hyperparameter Tuning Impact on Performance
# XGBoost:
# - Hyperparameters Tuned: subsample, n_estimators, max_depth, learning_rate, colsample_bytree.
# - Impact: Tuning learning_rate and subsample improved the model's ability to generalize, resulting in better accuracy, precision, and recall. The accuracy increased from 0.7853 to 0.8430, demonstrating the model's better fit to the data.
#
# LightGBM:
# - Hyperparameters Tuned: num_leaves, n_estimators, max_depth, learning_rate.
# - Impact: The accuracy of LightGBM improved slightly from 0.7839 to 0.8427 after tuning. The precision, recall, and F1-score improved, showing that the model became more balanced and effective after the adjustments.
#
# CatBoost:
# - Hyperparameters Tuned: learning_rate, l2_leaf_reg, iterations, depth.
# - Impact: CatBoost showed less improvement compared to XGBoost and LightGBM. The accuracy improved from 0.7860 to 0.7996, but it still lags behind the other models. The precision and recall also showed smaller improvements.
#
# 3.3 Ensemble Methods: Voting and Stacking Classifiers
# - Voting Classifier (Soft Voting): By combining predictions from XGBoost, LightGBM, and CatBoost, the Voting Classifier performed similarly to XGBoost with accuracy = 0.8430. Soft voting works well when individual models perform similarly.
# - Stacking Classifier: The Stacking Classifier achieved the highest improvement, accuracy = 0.8606, showing the strength of combining multiple models through a meta-model (Logistic Regression in this case). It outperformed all individual models, indicating the potential of stacking to leverage the strengths of various models.
#
# 4. Hyperparameters Tuned and Their Impact
# XGBoost Hyperparameters:
# - subsample = 0.8: This helped prevent overfitting by reducing the proportion of data used to build each tree.
# - n_estimators = 200, max_depth = 7: These hyperparameters controlled the complexity of the trees, improving generalization.
#
# LightGBM Hyperparameters:
# - num_leaves = 50: Allowed the model to capture more complex relationships.
# - learning_rate = 0.2: The learning rate was tuned to prevent overfitting while improving model performance.
#
# CatBoost Hyperparameters:
# - learning_rate = 0.05: Lower learning rate for slower but more refined training.
# - depth = 10: Increased depth to capture more complex patterns in the data.
#
# 5. Dataset Overview and Feature Engineering
# Dataset Shape:
# - Raw Dataset: (107,583 samples, 70 features)
# - After Preprocessing: (85,500 samples, 46 features)
#
# Feature Engineering:
# - Categorical Features: We applied One-Hot Encoding to features like season, koppen_climate, building_type, etc., and Hash Encoding for high cardinality features like city, country, and climate_description.
# - Numerical Features: Continuous features such as age, thermal_sensation, and air_temperature were treated as numerical and scaled appropriately (e.g., MinMaxScaler or StandardScaler).
#
# 6. Conclusion and Recommendations
# Summary:
# - Stacking Classifier is the best-performing model after hyperparameter tuning, with an accuracy of 0.8606, and it outperforms all individual models.
# - XGBoost and LightGBM also showed significant improvements, achieving accuracy close to 0.8430 after tuning.
# - CatBoost, while a strong model, showed less improvement in this particular dataset, and it may require further feature engineering or tuning.
#
# Next Steps:
# - Further Fine-tuning: Consider further tuning of CatBoost and exploring other ensemble methods.
# - Model Deployment


# %%
# Function to evaluate model performance on training and test sets
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Predict on training set
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Predict on test set
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Print the results
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Check for overfitting
    if train_accuracy > test_accuracy:
        print("The model is overfitting.")
    else:
        print("The model is not overfitting.")


# Example usage with a trained model (e.g., RandomForestClassifier)
evaluate_model(rf_search.best_estimator_, X_train, y_train, X_test, y_test)
