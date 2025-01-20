## Table of Contents
1. [Explainability Methods](#1-explainability-methods)
2. [Gradient Boosting](#2-gradient-boosting)
3. [DATA103 Project: Project Notebook](#3-data103-project-project-notebook)
---

## 1. Explainability Methods

### Introduction
This notebook explores various methods for explaining machine learning models. The focus is on post-hoc interpretability techniques and their practical implementation.

```
from sklearn.inspection import permutation_importance
```

### Chapter 1: Overview of Explainability
- Importance of interpretability in machine learning.
- Key techniques: SHAP, LIME, and Partial Dependence Plots (PDP).

```
import shap
explainer = shap.Explainer(model, data)
shap_values = explainer(data)
```

### Chapter 2: Implementation of Techniques
- SHAP: Visualizing feature contributions.
- LIME: Local interpretability through perturbation.
- PDP: Examining feature impacts globally.

```
# SHAP Visualization
shap.summary_plot(shap_values, data)
```

---

## 2. Gradient Boosting

### Introduction
This notebook delves into gradient boosting techniques, focusing on implementation and performance tuning.

```
from sklearn.ensemble import GradientBoostingClassifier
```

### Chapter 1: Understanding Gradient Boosting
- Theoretical foundation of boosting.
- Comparison with other ensemble methods like bagging.

```
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
```

### Chapter 2: Hyperparameter Tuning
- Learning rate, number of estimators, and maximum depth.
- Impact of hyperparameters on model performance.

```
from sklearn.model_selection import GridSearchCV
param_grid = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]}
grid = GridSearchCV(model, param_grid, cv=3)
grid.fit(X_train, y_train)
```

---

## 3. DATA103 Project: Project Notebook

### Introduction
This project explores advanced data analysis and visualization techniques using Python. The focus is on uncovering insights through data preprocessing, statistical analysis, and creating dynamic dashboards. The notebook integrates key libraries to handle data, perform computations, and build interactive visualizations.

### Prerequisites
Before running this notebook, ensure you have the following installed:

- Python 3.7+  
- Jupyter Notebook or JupyterLab  
- Key Python libraries:
  - Numpy (`pip install numpy`)  
  - Pandas (`pip install pandas`)  
  - Matplotlib (`pip install matplotlib`)  
  - Seaborn (`pip install seaborn`)  
  - Plotly (`pip install plotly`)  
  - Dash (`pip install dash`)  
  - Scipy (`pip install scipy`)  
  - Dash Bootstrap Components (`pip install dash-bootstrap-components`)  

Additionally, ensure the required datasets are downloaded and accessible in the working directory. The project uses the following dataset:

- [Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

### Features
1. **Data Loading and Preprocessing:**  
   - Handles missing values and outliers.  
   - Transforms data for optimal usability.  
   - Supports various file formats, including `.csv` and `.parquet`.  

2. **Statistical Analysis:**  
   - Performs correlation and descriptive statistics.  
   - Offers visual insights with correlation matrices.  

3. **Dynamic Visualizations:**  
   - Interactive line, bar, and scatter plots.  
   - Heatmaps to explore data distributions.  

4. **Dashboard Integration:**  
   - Filters for dynamic data exploration (e.g., date, category, and numerical ranges).  
   - Visualizes trends and patterns in an accessible format.  

5. **Advanced Features (if applicable):**  
   - Spatial analysis using geolocation data.  
   - Temporal insights with time-series analysis.  

### How to Run
1. Clone the repository or download the notebook file.  
2. Ensure dependencies are installed by running:  
   ```bash
   pip install numpy pandas matplotlib seaborn plotly dash scipy dash-bootstrap-components
   ```  
3. Launch Jupyter Notebook and open the `.ipynb` file:  
   ```bash
   jupyter notebook
   ```  
4. Follow the steps in the notebook to process and analyze the data.  
5. If running the dashboard, execute the respective script or cell to deploy the app.  

### Notes
- Dataset paths must be correctly defined in the notebook.  
- Ensure you have sufficient computational resources for large datasets.  

### Limitations
- Predictive modeling is not included but can be extended in future iterations.  
- Some advanced visualizations may require tweaking depending on dataset specifics.  

---

For any questions or collaboration opportunities, please contact Matthew Cuaresma at matthew_cuaresma@dlsu.edu.ph
