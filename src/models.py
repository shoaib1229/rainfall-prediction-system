"""
Models Module for Rainfall Prediction System.
Centralizes the construction of all model pipelines and tuned configurations.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def get_logistic_regression_baseline(random_state=42):
    """
    Constructs the Phase 2 baseline Logistic Regression pipeline:
    SMOTE + StandardScaler + LogisticRegression(C=1.0, penalty='l2').
    """
    return ImbPipeline([
        ('smote',  SMOTE(random_state=random_state)),
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(C=1.0, penalty='l2', random_state=random_state, max_iter=1000))
    ])

def get_logistic_regression_tuned(random_state=42):
    """
    Constructs the Phase 3 tuned Logistic Regression pipeline:
    SMOTE + StandardScaler + LogisticRegression(C=0.1, penalty='l1', solver='liblinear').
    """
    return ImbPipeline([
        ('smote',  SMOTE(random_state=random_state)),
        ('scaler', StandardScaler()),
        ('clf',    LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=random_state, max_iter=2000))
    ])

def get_decision_tree_tuned(random_state=42):
    """
    Constructs the Phase 3 tuned Decision Tree classifier:
    max_depth=3, min_samples_leaf=4, min_samples_split=2, criterion='entropy'.
    """
    return DecisionTreeClassifier(
        criterion='entropy',
        max_depth=3,
        min_samples_leaf=4,
        min_samples_split=2,
        class_weight=None,
        random_state=random_state
    )

def get_random_forest_tuned(random_state=42):
    """
    Constructs the Phase 3 tuned Random Forest classifier:
    n_estimators=150, max_depth=6, min_samples_leaf=2, min_samples_split=2, max_features='sqrt'.
    """
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        min_samples_leaf=2,
        min_samples_split=2,
        max_features='sqrt',
        class_weight=None,
        random_state=random_state
    )

def get_all_models(random_state=42):
    """
    Returns a structured dictionary of all candidate models for evaluation.
    """
    return {
        'Logistic Regression (Baseline)': {
            'model': get_logistic_regression_baseline(random_state),
            'balancing': 'SMOTE (ImbPipeline)',
            'params': 'C=1.0, penalty=l2, default'
        },
        'Logistic Regression (Tuned)': {
            'model': get_logistic_regression_tuned(random_state),
            'balancing': 'SMOTE (ImbPipeline)',
            'params': 'C=0.1, penalty=l1, liblinear'
        },
        'Decision Tree (Tuned)': {
            'model': get_decision_tree_tuned(random_state),
            'balancing': 'class_weight=None',
            'params': 'max_depth=3, min_leaf=4, criterion=entropy'
        },
        'Random Forest (Tuned)': {
            'model': get_random_forest_tuned(random_state),
            'balancing': 'class_weight=None',
            'params': 'n_est=150, max_depth=6, min_leaf=2, sqrt'
        }
    }
