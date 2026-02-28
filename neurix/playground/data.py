"""
neurix/playground/data.py
Constants for the playground feature.
Questions are no longer hardcoded — they are AI-generated per match
based on the common modules both players have completed.
"""

TOTAL_ROUNDS     = 10
POINTS_PER_ROUND = 1
DISCONNECT_BONUS = 3
ROUND_TIMEOUT    = 30      # seconds before a round is declared no-winner
LABELS           = ["A", "B", "C", "D"]

# Module metadata used to build the AI prompt.
# Maps module_id → human-readable title and topic description.
MODULE_TOPICS = {
    "b01_what_is_ml": {
        "title": "What is Machine Learning?",
        "description": "supervised vs unsupervised vs reinforcement learning, traditional programming vs ML, how models learn from data",
    },
    "b02_python_numpy": {
        "title": "Python & NumPy Basics",
        "description": "NumPy ndarray, vectorised operations, broadcasting, array shape and dtype, boolean indexing, linspace, arange",
    },
    "b03_linear_regression": {
        "title": "Linear Regression",
        "description": "linear regression equation, weights and bias, Mean Squared Error, fitting a line to data, scikit-learn LinearRegression",
    },
    "b04_data_preprocessing": {
        "title": "Data Preprocessing",
        "description": "handling missing values, StandardScaler, train/test split, one-hot encoding, label encoding, data leakage",
    },
    "i01_logistic_regression": {
        "title": "Logistic Regression",
        "description": "sigmoid function, binary classification, decision boundary, binary cross-entropy loss, probability output",
    },
    "i02_decision_trees": {
        "title": "Decision Trees & Random Forests",
        "description": "Gini impurity, information gain, bagging, random feature subsets, max_depth, n_estimators, overfitting in trees",
    },
    "i03_neural_networks": {
        "title": "Neural Networks",
        "description": "feedforward networks, hidden layers, activation functions (ReLU, sigmoid, softmax, tanh), backpropagation, loss functions",
    },
    "i04_sql_for_ml": {
        "title": "SQL for ML Data Pipelines",
        "description": "SELECT, WHERE, GROUP BY, HAVING, JOIN, window functions, aggregation for ML feature engineering",
    },
    "a01_cnn": {
        "title": "Convolutional Neural Networks",
        "description": "Conv2D, MaxPooling, Flatten, Dropout, receptive field, spatial locality, parameter sharing, feature maps",
    },
    "a02_transformers": {
        "title": "Transformers & Attention",
        "description": "self-attention, Query/Key/Value, multi-head attention, positional encoding, scaled dot-product attention, why transformers beat RNNs",
    },
    "a03_js_data_viz": {
        "title": "Data Visualisation with JavaScript",
        "description": "Canvas API, decision boundary visualisation, scatter plots, understanding model output visually",
    },
}
