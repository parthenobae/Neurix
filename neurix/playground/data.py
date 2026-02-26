
# Each question has: question, answer (correct), distractors (3 wrong options)
QUESTIONS = [
    {
        "question": "Which algorithm is commonly used for binary classification and outputs probabilities using a sigmoid function?",
        "answer": "Logistic Regression",
        "distractors": ["Linear Regression", "Decision Tree", "K-Nearest Neighbors"],
    },
    {
        "question": "What does overfitting mean in machine learning?",
        "answer": "The model memorises training data and fails to generalise",
        "distractors": [
            "The model is too simple to capture patterns",
            "The model converges too slowly",
            "The model uses too few features",
        ],
    },
    {
        "question": "Which validation strategy repeatedly splits data into training and validation folds?",
        "answer": "K-Fold Cross Validation",
        "distractors": ["Hold-Out Validation", "Leave-One-Out", "Bootstrap Sampling"],
    },
    {
        "question": "In gradient descent, what parameter controls the step size of each update?",
        "answer": "Learning Rate",
        "distractors": ["Momentum", "Batch Size", "Weight Decay"],
    },
    {
        "question": "Which metric is often preferred over accuracy for imbalanced binary datasets?",
        "answer": "F1 Score",
        "distractors": ["Mean Squared Error", "R² Score", "Log Loss"],
    },
    {
        "question": "What type of neural network layer connects every neuron to every neuron in the next layer?",
        "answer": "Fully Connected (Dense) Layer",
        "distractors": ["Convolutional Layer", "Pooling Layer", "Recurrent Layer"],
    },
    {
        "question": "Which technique randomly drops neurons during training to reduce overfitting?",
        "answer": "Dropout",
        "distractors": ["Batch Normalisation", "L2 Regularisation", "Early Stopping"],
    },
    {
        "question": "What is the name of the process of adjusting model weights using the chain rule of calculus?",
        "answer": "Backpropagation",
        "distractors": ["Forward Pass", "Gradient Clipping", "Weight Initialisation"],
    },
    {
        "question": "Which unsupervised learning algorithm groups data points into k clusters?",
        "answer": "K-Means Clustering",
        "distractors": ["DBSCAN", "Principal Component Analysis", "Linear Discriminant Analysis"],
    },
    {
        "question": "Which algorithm builds an ensemble of decision trees using random feature subsets?",
        "answer": "Random Forest",
        "distractors": ["AdaBoost", "Support Vector Machine", "Naive Bayes"],
    },
    {
        "question": "What term describes the error caused by overly simple assumptions in a learning algorithm?",
        "answer": "Bias",
        "distractors": ["Variance", "Entropy", "Regularisation"],
    },
    {
        "question": "Which activation function outputs values strictly between 0 and 1, used in binary classification output layers?",
        "answer": "Sigmoid",
        "distractors": ["ReLU", "Tanh", "Softmax"],
    },
    {
        "question": "What is the name of the optimisation algorithm that adapts learning rates for each parameter individually?",
        "answer": "Adam",
        "distractors": ["SGD", "RMSProp", "Adagrad"],
    },
    {
        "question": "Which dimensionality reduction technique projects data onto directions of maximum variance?",
        "answer": "Principal Component Analysis (PCA)",
        "distractors": ["t-SNE", "UMAP", "Linear Discriminant Analysis"],
    },
    {
        "question": "What is the purpose of a confusion matrix in classification tasks?",
        "answer": "To show the counts of true/false positives and negatives",
        "distractors": [
            "To measure the distance between class centroids",
            "To visualise feature correlations",
            "To plot the learning curve",
        ],
    },
    {
        "question": "Which loss function is standard for multi-class classification with softmax output?",
        "answer": "Categorical Cross-Entropy",
        "distractors": ["Mean Squared Error", "Hinge Loss", "Huber Loss"],
    },
    {
        "question": "What does the ROC curve plot?",
        "answer": "True Positive Rate vs False Positive Rate",
        "distractors": [
            "Precision vs Recall",
            "Loss vs Epochs",
            "Accuracy vs Model Complexity",
        ],
    },
    {
        "question": "Which technique scales each feature to have zero mean and unit variance?",
        "answer": "Standardisation (Z-score normalisation)",
        "distractors": ["Min-Max Scaling", "Log Transformation", "One-Hot Encoding"],
    },
]

TOTAL_ROUNDS     = 10
POINTS_PER_ROUND = 1
DISCONNECT_BONUS = 3
LABELS = ["A", "B", "C", "D"]

