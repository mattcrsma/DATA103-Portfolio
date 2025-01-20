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
The project notebook consolidates the knowledge from the previous notebooks and applies it to a real-world dataset.

```
import pandas as pd
data = pd.read_csv('dataset.csv')
```

### Chapter 1: Data Exploration
- Overview of the dataset: Summary statistics and visualization.
- Handling missing data and outliers.

```
data.describe()
```

### Chapter 2: Model Development
- Feature engineering and selection.
- Model training and evaluation using explainability methods.

```
from sklearn.metrics import accuracy_score
accuracy_score(y_test, model.predict(X_test))
```

### Chapter 3: Results and Insights
- Performance metrics and visualizations.
- Interpretation of results using SHAP and LIME.

```
# SHAP-based insight
shap.waterfall_plot(shap.Explanation(values, base_values, data))
```

---

For any questions or collaboration opportunities, please contact Matthew Cuaresma at matthew_cuaresma@dlsu.edu.ph
