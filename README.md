# MWS-Project
Using Genetic Algorithms to Enhance Machine Learning for Medical Data Analysis

1. Project Description

This project explores the use of Genetic Algorithms (GA) to enhance machine learning models applied to medical data analysis, with a focus on feature selection for high-dimensional datasets.

Medical data often contains a large number of features, which may reduce interpretability and increase computational cost. To address this issue, GA is employed to select the most informative features while preserving competitive classification performance.

Breast cancer diagnosis is used as a case study to validate the proposed approach.

2. Methodology Summary

The methodology consists of three main components:

Baseline Models: Logistic Regression (LR), Support Vector Machine (SVM), and Random Forest (RF) trained using all features as reference benchmarks.

Genetic Algorithm (GA): Applied as a wrapper-based feature selection method using two variants:

    GA Basic: Fitness based on test-set accuracy with a penalty on feature count.

    GA Improved: Fitness based on Accuracy and F1-macro with internal cross-validation for improved robustness.

Evaluation:
Models were assessed using Accuracy, F1-macro, ROC-AUC and Confusion Matrix, with statistical tests used to assess performance differences.

3. Case Study

The framework was evaluated using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset:

    569 samples

    30 numerical features

    Binary classification: Benign / Malignant

4. Interactive Application

An interactive application was developed using Streamlit and SQLite to demonstrate practical applicability.

The interface allows users to:

    Enter or upload patient data (CSV batch upload)

    Manage data using CRUD operations

    Run baseline models and GA-based feature selection

    Adjust GA hyperparameters

    Visualize model performance and selected features

5. Project Structure

Experimental Scripts (Method Validation):

    00_baseline_all_features.py

    01_ga_basic.py

    02_ga_improved_cv_plots.py

Interactive Application (Interface-final):

    app.py

    genetic_algorithm.py

    utils.py

    create_db.py

    patients.db

6. How to Run the Project:

6.1 Requirements

    Python 3.9 or higher

    Required libraries:

    pip install numpy pandas scikit-learn deap streamlit matplotlib seaborn

6.2 Run Experimental Scripts (Optional)

To reproduce the experimental results:

    python 00_baseline_all_features.py
    python 01_ga_basic.py
    python 02_ga_improved_cv_plots.py

6.3 Run the Interactive Application

Navigate to the interface folder:

    cd Interface-final

    (Optional) Initialize the database:

    python create_db.py

Launch the application:

    streamlit run app.py

Open the provided local URL in your browser to interact with the application.

7. Academic Context

This project was developed as part of a Master’s thesis on applying evolutionary optimization techniques to enhance machine learning for medical data analysis.