# ACM_autojudge
# AutoJudge – Predicting Programming Problem Difficulty

## Project Overview
AutoJudge is a machine learning-based system that automatically predicts the
difficulty of programming problems using only textual information. The system
performs both classification (Easy / Medium / Hard) and regression (numerical
difficulty score) and provides predictions through a simple web interface.

## Dataset Used
We used the TaskComplexityEval-24 dataset, which contains programming problems
along with difficulty labels and numerical difficulty scores. Each sample
includes:
- Title
- Problem description
- Input description
- Output description
- Difficulty class
- Difficulty score

Dataset source:
https://github.com/AREEG94FAHAD/TaskComplexityEval-24

## Approach
1. Combined all textual fields into a single input.
2. Applied text preprocessing and feature engineering.
3. Extracted features using TF-IDF along with engineered numerical features.
4. Train:
   - Classification model for difficulty class
   - Regression model for difficulty score
5. Deployed models using a Streamlit-based web interface.

## Models Used
- Classification: Linear Support Vector Machine (SVM)
- Regression: Random Forest Regressor

## Evaluation Metrics
- Classification Accuracy
- Confusion Matrix
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

## Web Interface
The web interface is implemented using Streamlit. Users can input a programming
problem description along with input and output formats, and the system predicts
the difficulty class and difficulty score in real time.

## Steps to Run Locally
1. Installed dependencies:
   pip install streamlit scikit-learn scipy joblib numpy

2. Navigated to the project folder.

3. Run the web application:
   streamlit run src/app.py

4. Opened the browser at http://localhost:8501


