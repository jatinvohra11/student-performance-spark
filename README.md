# 🎓 Student Performance Prediction & Analysis Using Apache Spark MLlib

<p align="center">
  <strong>A Big Data Analytics and Machine Learning Project</strong>
</p>

<p align="center">
  Analyzing student academic performance, predicting final grades, classifying pass/fail outcomes, and discovering student groups using Apache Spark and PySpark MLlib.
</p>

---

## 📌 1. Project Overview

Student Performance Prediction & Analysis is a data analytics and machine learning project developed using **Apache Spark, PySpark, and Spark MLlib**.

The project uses student academic and demographic information to understand performance patterns, identify factors associated with final grades, predict students' final grades, classify pass/fail outcomes, and group students according to similarities in their academic attributes.

The project uses the Student Performance dataset from the UCI Machine Learning Repository, specifically the Portuguese language subject dataset (`student-por.csv`).

The complete workflow includes data loading, data quality inspection, exploratory data analysis, visualization, machine learning model training, model evaluation, student clustering, and exporting results to CSV files.

### ✨ Key Highlights

- Distributed data processing with Apache Spark.
- Exploratory Data Analysis (EDA) using PySpark and Python visualization libraries.
- Final grade prediction using Linear Regression and Decision Tree Regression.
- Pass/fail classification using Logistic Regression.
- Student segmentation using the K-Means clustering algorithm.
- Evaluation using R², RMSE, MAE, accuracy, precision, recall, and F1-score.
- Correlation heatmaps and model comparison charts.
- Export of model evaluation results to CSV files.

---

## 🎯 2. Problem Statement

Educational institutions generate large amounts of student data, including demographic details, study habits, attendance-related information, previous grades, and academic outcomes.

Analyzing this data manually can make it difficult to identify performance patterns, understand relationships between academic variables, and estimate future student outcomes.

The problem addressed by this project is:

**How can big data processing and machine learning techniques be used to analyze student academic performance, predict final grades, classify pass/fail outcomes, and discover groups of students with similar academic characteristics?**

This project addresses the problem by building a data analysis and machine learning workflow using Apache Spark and PySpark MLlib.

---

## 🎯 3. Project Objectives

1. Load and process student data using Apache Spark.
2. Inspect dataset structure, column names, data types, duplicate records, and missing values.
3. Perform exploratory data analysis on student grades.
4. Visualize the distribution of final grades.
5. Study correlations between academic attributes using a heatmap.
6. Predict final grades using regression algorithms.
7. Classify students into pass and fail categories.
8. Evaluate machine learning models using suitable performance metrics.
9. Group students with similar characteristics using K-Means clustering.
10. Compare model performance and export results for further analysis.

---

## 📂 4. Dataset Description

**Dataset:** Student Performance Dataset  
**Source:** UCI Machine Learning Repository  
**Subject:** Portuguese language  
**File:** `student-por.csv`

The dataset contains **649 student records and 33 columns**.

Dataset source: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

### Important Features

| Feature | Description |
|---|---|
| `school` | School identifier |
| `sex` | Student's recorded sex |
| `age` | Student's age |
| `address` | Urban or rural residence category |
| `famsize` | Family size category |
| `Pstatus` | Parents' cohabitation status |
| `Medu` | Mother's education level |
| `Fedu` | Father's education level |
| `studytime` | Weekly study-time category |
| `failures` | Number of previous class failures |
| `absences` | Number of school absences |
| `G1` | First-period grade |
| `G2` | Second-period grade |
| `G3` | Final grade; the main regression target |

The original dataset also contains additional family, school, social, and behavioral attributes.

**Important:** The regression model in this implementation uses `age`, `studytime`, `failures`, `absences`, `G1`, and `G2` as input features and `G3` as the target. The target `G3` is not included in the input feature vector.

---

## 🛠️ 5. Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.10 | Main programming language |
| Apache Spark | Distributed data processing framework |
| PySpark | Python interface for Apache Spark |
| Spark SQL | DataFrame operations, filtering, grouping, and aggregation |
| Spark MLlib | Machine learning model training and evaluation |
| VectorAssembler | Combines selected columns into a feature vector |
| Matplotlib | Creates bar charts and model comparison visualizations |
| Seaborn | Creates correlation heatmaps |
| Pandas | Converts selected results into tabular form for visualization and CSV export |
| Jupyter Notebook | Interactive development and experimentation |
| VS Code | Code editing and project development |
| Git and GitHub | Version control and project hosting |

### What Does Each Technology Do?

**Apache Spark:** Processes the dataset through Spark's DataFrame API. The application is configured in local mode using `local[*]`, allowing Spark to use the available logical CPU cores.

**PySpark:** Makes Spark functionality accessible through Python code.

**Spark SQL:** Provides operations such as `select()`, `groupBy()`, `count()`, `describe()`, and `agg()` to inspect and analyze student records.

**Spark MLlib:** Provides the regression, classification, clustering, feature transformation, and evaluation components used in the project.

**Matplotlib and Seaborn:** Convert numerical analysis results into charts that are easier to interpret.

**Pandas:** Supports presentation of selected results and saving result tables as CSV files.

---

## ⚙️ 6. Project Workflow

The project follows this pipeline:

1. Initialize SparkSession.
2. Load the student dataset.
3. Inspect the schema and data quality.
4. Calculate grade statistics.
5. Visualize grade distribution.
6. Calculate correlations and generate a heatmap.
7. Train and evaluate regression models.
8. Train and evaluate the pass/fail classifier.
9. Perform K-Means clustering.
10. Compare results and export evaluation tables.

---

## 🔍 7. Exploratory Data Analysis (EDA)

Exploratory Data Analysis is used to understand the structure, distribution, and relationships within the dataset.

### 7.1 Dataset Inspection

The dataset is inspected using Spark DataFrame operations.

The analysis includes:

- Total record count.
- Total number of columns.
- Column names and data types.
- Dataset schema.
- Sample student records.
- Duplicate record count.
- Missing-value count by column.

This stage helps understand the dataset before performing statistical analysis or training machine learning models.

### 7.2 Grade Statistics

Descriptive statistics are calculated for `G1`, `G2`, and `G3`.

The analysis includes:

- Number of observations.
- Mean grade.
- Standard deviation.
- Minimum grade.
- Maximum grade.

The observed average final grade (`G3`) is approximately **11.91 out of 20**, with a minimum of 0 and a maximum of 19.

### 7.3 Final Grade Distribution

**Visualization: Bar Chart — Distribution of Student Final Grades**

The bar chart displays final grades on the X-axis and the number of students receiving each grade on the Y-axis.

**Purpose:**
- Identify frequently occurring grades.
- Understand how student grades are distributed.
- Observe whether the dataset contains low, middle, or high final grades.
- Identify grade values that occur less frequently.

The chart helps summarize overall academic performance visually.

### 7.4 Correlation Heatmap

**Visualization: Correlation Heatmap — Student Performance**

The heatmap represents Pearson correlation coefficients between selected numerical attributes.

The analysis includes:

- Age.
- Study time.
- Previous failures.
- Absences.
- First-period grade (`G1`).
- Second-period grade (`G2`).
- Final grade (`G3`).

Correlation values range from -1 to +1.

- **Positive correlation:** Two variables tend to increase together.
- **Negative correlation:** Higher values of one variable tend to be associated with lower values of another.
- **Correlation near zero:** Little or no linear relationship is observed.

In the observed results, `G1` and `G3` have a correlation of approximately **0.83**, while `G2` and `G3` have a correlation of approximately **0.92**.

These values indicate strong positive relationships between previous grades and final grades in this dataset. Correlation does not establish causation.

### 7.5 Final Grade Summary and Pass/Fail Distribution

The project calculates summary statistics for `G3` and counts students using the rule:

- **Pass:** `G3 >= 10`
- **Fail:** `G3 < 10`

Observed counts:

| Category | Number of Students |
|---|---:|
| Pass | 549 |
| Fail | 100 |
| Total | 649 |

This threshold is a project-defined classification rule for the analysis.

---

## 🤖 8. Machine Learning Models

The project implements regression, classification, and clustering algorithms using Spark MLlib.

### 8.1 Linear Regression — Final Grade Prediction

**Objective:** Predict the final student grade (`G3`) from selected student attributes.

**Input features:**
- `age`
- `studytime`
- `failures`
- `absences`
- `G1`
- `G2`

**Target:** `G3`

#### Implementation Process

1. Select the input features and target column.
2. Split the dataset into training and testing subsets.
3. Use `VectorAssembler` to combine input columns into a feature vector.
4. Train the Linear Regression model on the training data.
5. Generate predictions for the test data.
6. Evaluate prediction quality using R², RMSE, and MAE.

The model uses the following configuration:

- Maximum iterations: 100
- Regularization parameter: 0.01
- Random split seed: 42

The dataset split produces 544 training records and 105 testing records in the current execution.

#### Evaluation Metrics

**R² Score:** Measures how well the model explains variation in the target variable. A higher score generally indicates a better fit.

**RMSE (Root Mean Squared Error):** Measures prediction error while giving greater weight to larger errors. Lower values are better.

**MAE (Mean Absolute Error):** Measures the average absolute difference between actual and predicted grades. Lower values are better.

#### Observed Results

| Metric | Linear Regression |
|---|---:|
| R² Score | 0.8495 |
| RMSE | 1.1764 |
| MAE | 0.7250 |

These results are from the current train-test split and model configuration.

### 8.2 Decision Tree Regression

**Objective:** Predict final grades using a tree-based regression algorithm.

Decision Tree Regression splits the data into decision-based regions and predicts numerical values from the learned tree structure.

#### Implementation Process

1. Reuse the prepared training and testing feature vectors.
2. Configure `DecisionTreeRegressor`.
3. Set the maximum tree depth to 5.
4. Train the model.
5. Generate predictions for the test data.
6. Evaluate using R², RMSE, and MAE.

The model uses the same selected input features and target as the Linear Regression model.

#### Observed Results

| Metric | Decision Tree Regression |
|---|---:|
| R² Score | 0.8764 |
| RMSE | 1.0662 |
| MAE | 0.7450 |

The Decision Tree model obtains a higher R² score and lower RMSE than Linear Regression in this run. However, Linear Regression has the lower MAE.

The preferred model therefore depends on the evaluation metric and the intended use.

### 8.3 Regression Model Comparison

**Visualization: Regression Model Comparison**

A bar chart compares the R² scores of Linear Regression and Decision Tree Regression.

The chart helps identify which model explains more variation in final grades on the current test split.

A separate results table presents R², RMSE, and MAE side by side for both models.

This makes it easier to compare the models using multiple evaluation criteria rather than relying on a single score.

---

## 🎯 9. Pass/Fail Classification Using Logistic Regression

### Objective

Classify students into pass and fail categories using a threshold of 10 marks on the final grade.

The target label is generated using:

- `1` when `G3 >= 10`
- `0` when `G3 < 10`

**Input features:** `age`, `studytime`, `failures`, `absences`, `G1`, and `G2`.

**Target:** `pass_label`

### Implementation Process

1. Create the pass/fail label.
2. Prepare the classification feature vector.
3. Split the data into training and testing subsets.
4. Train a Spark MLlib Logistic Regression classifier.
5. Generate predicted labels and class probabilities.
6. Calculate accuracy, weighted precision, weighted recall, and weighted F1-score.
7. Generate a confusion matrix.

The classification workflow uses the same 80:20 split ratio and seed of 42.

### 9.1 Classification Metrics

**Accuracy:** The proportion of test examples classified correctly.

**Precision:** Measures how many predictions assigned to a class are correct.

**Recall:** Measures how many actual examples of a class are identified correctly.

**F1-score:** The harmonic mean of precision and recall.

Weighted precision, recall, and F1-score account for the support of each class when aggregating the class-level metrics.

### Observed Results

| Metric | Observed Score |
|---|---:|
| Accuracy | 99.05% |
| Weighted Precision | 0.9912 |
| Weighted Recall | 0.9905 |
| Weighted F1-score | 0.9906 |

**Important interpretation:** These high scores should be treated cautiously. The current classifier uses `G1` and `G2`, which are strongly associated with `G3`. The classification label is derived from `G3`, so these results reflect a prediction setting where earlier grades are already available. The results do not guarantee similar performance on other datasets or future students.

### 9.2 Confusion Matrix

The confusion matrix compares actual pass/fail labels with predicted labels.

For binary classification:

- True Positive (TP): Actual pass predicted as pass.
- True Negative (TN): Actual fail predicted as fail.
- False Positive (FP): Actual fail predicted as pass.
- False Negative (FN): Actual pass predicted as fail.

The confusion matrix helps identify the types of classification errors, which overall accuracy alone may hide.

### 9.3 Classification Performance Chart

**Visualization: Logistic Regression Performance**

A bar chart displays Accuracy, Precision, Recall, and F1-score.

The chart provides a quick visual comparison of the classification metrics. Accuracy is displayed as a percentage, while the other reported metrics are expressed on a 0–1 scale in the current implementation; both are normalized to a similar range for visualization.

---

## 🔵 10. Student Clustering Using K-Means

### Objective

Discover groups of students with similar academic characteristics without requiring predefined class labels.

K-Means is an unsupervised learning algorithm that assigns records to clusters based on distances in feature space.

### Features Used

- `age`
- `studytime`
- `failures`
- `absences`
- `G1`
- `G2`

The final grade `G3` is not included in the clustering feature vector.

### Implementation Process

1. Select the numerical clustering features.
2. Combine the features using `VectorAssembler`.
3. Configure the K-Means model with `k = 3`.
4. Train the model.
5. Assign each student a cluster label.
6. Count the number of students in each cluster.
7. Calculate cluster-level averages.
8. Visualize the average final grade for each cluster.

### 10.1 Cluster Distribution

**Visualization: Student Cluster Counts**

The cluster distribution table shows how many students are assigned to each cluster.

Observed cluster sizes:

| Cluster Label | Students |
|---|---:|
| Cluster 1 | 263 |
| Cluster 2 | 278 |
| Cluster 0 | 108 |
| **Total** | **649** |

Cluster labels are identifiers, not rankings. Cluster 0 is not automatically worse than Cluster 1 or Cluster 2.

### 10.2 Cluster Analysis

The project calculates cluster averages for:

- First-period grade (`G1`).
- Second-period grade (`G2`).
- Study time.
- Absences.
- Previous failures.

These summaries help describe the academic characteristics of each group.

Clusters may represent different patterns of academic performance and study-related attributes. Their meanings must be interpreted from the observed feature averages rather than assumed in advance.

### 10.3 Average Grade Across Clusters

**Visualization: Average Final Grade Across Student Clusters**

The bar chart displays each cluster on the X-axis and its average final grade (`G3`) on the Y-axis.

In the current results, Cluster 1 has the highest average final grade, followed by Cluster 0 and Cluster 2.

The chart helps compare the final-grade distributions represented by the clusters. Since `G3` is not used to train K-Means, it is used here to interpret the clusters after training.

---

## 📊 11. Final Results and Export

The project organizes model evaluation results into tabular form for comparison and reuse.

### Regression Results

The regression results table contains:

- Model name.
- R² score.
- RMSE.
- MAE.

### Classification Results

The classification results table contains:

- Accuracy.
- Weighted precision.
- Weighted recall.
- Weighted F1-score.

### Output Files

The notebook exports the results to:

- `regression_results.csv`
- `classification_results.csv`

These files can be opened in spreadsheet applications or imported into other analysis tools.

---

## 📁 12. Project Structure

```text
student-performance-spark/
│
├── data/
│   └── student-por.csv
│
├── Student_Performance_Prediction.ipynb
├── regression_results.csv
├── classification_results.csv
├── README.md
└── .gitignore
```

The `.venv/` directory contains the local Python environment and should not be uploaded to GitHub.

---

## 💻 13. Installation and Setup

### Prerequisites

- Python 3.10
- Java runtime compatible with the installed PySpark version
- Git
- Visual Studio Code or Jupyter Notebook

This project was developed with Python 3.10 and PySpark 3.5.x. The exact Java compatibility should be checked against the installed Spark/PySpark release.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/student-performance-spark.git
cd student-performance-spark
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Create a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install pyspark==3.5.6 pandas matplotlib seaborn jupyter ipykernel
```

### Step 4: Download the Dataset

Download the Student Performance dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/320/student+performance).

Create the `data/` directory if it does not exist:

```bash
mkdir -p data
```

Place the Portuguese dataset file at:

```text
data/student-por.csv
```

### Step 5: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `Student_Performance_Prediction.ipynb` and select the Python environment containing PySpark.

### Step 6: Run the Notebook

Run the notebook cells in order, starting with SparkSession initialization and dataset loading.

The notebook will generate analysis outputs, visualizations, model predictions, evaluation metrics, and CSV result files.

---

## 📈 14. Visualizations Included

| Visualization | Purpose |
|---|---|
| Final Grade Distribution Bar Chart | Shows the number of students receiving each final grade |
| Correlation Heatmap | Shows relationships between numerical student attributes |
| Regression Model Comparison | Compares the R² scores of Linear Regression and Decision Tree Regression |
| Logistic Regression Performance Chart | Compares accuracy, precision, recall, and F1-score |
| Average Grade Across Clusters | Compares average final grades for the three K-Means clusters |

Together, these visualizations communicate grade patterns, relationships between variables, model performance, and student group characteristics.

---

## ⚠️ 15. Limitations and Future Improvements

### Current Limitations

- The project uses a single subject-specific dataset with 649 records.
- Results depend on the selected features, random split, and model configuration.
- Correlation does not prove causation.
- The pass/fail threshold of 10 is a project-defined rule.
- The current classification model uses earlier grades, which are strongly related to the final grade.
- The clustering features are used without a scaling step in the current implementation, so variables with larger numeric ranges may influence Euclidean distances more strongly.
- The current workflow runs in Spark local mode. It demonstrates Spark-based processing and MLlib but does not demonstrate execution across a multi-node cluster.
- The reported metrics are from one train-test split and are not a substitute for external validation.

### Future Improvements

1. Compare grade prediction using demographic and study-related features without `G1` and `G2`.
2. Use cross-validation and hyperparameter tuning.
3. Add Decision Tree and other suitable classification models.
4. Apply feature scaling before K-Means clustering and compare cluster quality.
5. Add confusion-matrix visualization and per-class classification metrics.
6. Test models on additional datasets or a separate external dataset.
7. Experiment with larger datasets and a real multi-node Spark cluster.
8. Build an interactive dashboard using Streamlit or Power BI.
9. Add model persistence so trained models can be reused.
10. Explore methods for identifying students who may benefit from additional academic support.

---

## 🎓 16. Learning Outcomes

This project demonstrates practical experience with:

- Big data processing concepts.
- Apache Spark and PySpark DataFrames.
- Exploratory Data Analysis.
- Feature engineering and vector assembly.
- Supervised machine learning.
- Regression and classification evaluation.
- Unsupervised learning using K-Means.
- Data visualization and result interpretation.
- CSV export and reproducible project organization.

---

## 🏁 17. Conclusion

The Student Performance Prediction & Analysis project demonstrates how Apache Spark and MLlib can be used to analyze student data through statistical summaries, visualizations, supervised machine learning, and unsupervised clustering.

Linear Regression and Decision Tree Regression are used to predict final grades, while Logistic Regression classifies students into pass/fail categories. K-Means discovers groups of students with similar academic characteristics.

The evaluation results and visualizations provide a structured way to explore student performance patterns. The project also highlights important considerations such as feature selection, evaluation metrics, data leakage risks, and the difference between local Spark execution and a distributed cluster deployment.

The project serves as a foundation for further work in educational analytics, big data processing, and machine learning.

---

## 📚 18. References

1. **UCI Machine Learning Repository — Student Performance Dataset**  
   https://archive.ics.uci.edu/dataset/320/student+performance

2. **Apache Spark Documentation**  
   https://spark.apache.org/docs/latest/

3. **PySpark Documentation**  
   https://spark.apache.org/docs/latest/api/python/

4. **Spark MLlib Guide**  
   https://spark.apache.org/docs/latest/ml-guide.html

5. **Matplotlib Documentation**  
   https://matplotlib.org/stable/

6. **Seaborn Documentation**  
   https://seaborn.pydata.org/

7. **Pandas Documentation**  
   https://pandas.pydata.org/docs/

---

## 👨‍💻 Author

**Jatin Vohra**

Project — Cluster Computing

**Project:** Student Performance Prediction & Analysis Using Apache Spark MLlib

---

⭐ If you find this project useful for learning big data analytics and machine learning, consider starring the repository.