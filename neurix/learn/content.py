"""
neurix/learn/content.py
All module content: theory, code challenges, MCQ questions.
"""
from __future__ import annotations
from typing import Dict, List, Any

# ── Module content registry ───────────────────────────────────────────────────
# Structure:
#   id         : unique slug used in DB + URLs
#   level      : "beginner" | "intermediate" | "advanced"
#   order      : display order within the level
#   title      : short title
#   description: one-line summary shown on dashboard card
#   theory     : HTML string rendered in module page
#   has_ide    : bool — show the code IDE?
#   ide_language: "python" | "javascript" | "sql" | None
#   ide_starter: starter code shown in editor
#   ide_solution_check: keyword(s) that must appear in output to pass
#   challenge_description: text shown above the IDE
#   question   : MCQ question dict (label, options) shown below theory
#   points     : points awarded on completion

MODULES: List[Dict[str, Any]] = [

    # ═══════════════════════════ BEGINNER ════════════════════════════════════

    {
        "id": "b01_what_is_ml",
        "level": "beginner",
        "order": 1,
        "title": "What is Machine Learning?",
        "description": "Understand what ML is, why it matters, and how it differs from traditional programming.",
        "points": 2,
        "theory": """
<p>Machine Learning (ML) is a subfield of Artificial Intelligence where systems learn from data
to make decisions or predictions — <strong>without being explicitly programmed for each task</strong>.</p>

<h5 class="mt-4">Traditional Programming vs Machine Learning</h5>
<div class="comparison-table">
  <table class="table table-bordered">
    <thead><tr><th>Traditional Programming</th><th>Machine Learning</th></tr></thead>
    <tbody>
      <tr><td>Rules + Data → Output</td><td>Data + Output → Rules</td></tr>
      <tr><td>You write every condition</td><td>The model learns conditions from examples</td></tr>
      <tr><td>Brittle with new input</td><td>Generalises to unseen data</td></tr>
    </tbody>
  </table>
</div>

<h5 class="mt-4">Three Types of Machine Learning</h5>
<ul>
  <li><strong>Supervised Learning</strong> — learn from labelled examples (e.g. spam detection)</li>
  <li><strong>Unsupervised Learning</strong> — find patterns in unlabelled data (e.g. customer segmentation)</li>
  <li><strong>Reinforcement Learning</strong> — learn by reward/punishment (e.g. game-playing agents)</li>
</ul>

<h5 class="mt-4">A Simple Mental Model</h5>
<p>Think of teaching a child to recognise cats. You show them thousands of photos labelled "cat" or "not cat".
They build an internal model. Later they recognise cats they've never seen before.
That's supervised machine learning.</p>
""",
        "has_ide": False,
        "question": {
            "text": "In Machine Learning, what do we provide to get the rules/model out?",
            "options": [
                {"label": "A", "text": "Rules and conditions written by hand", "correct": False},
                {"label": "B", "text": "Data and desired outputs (labels)", "correct": True},
                {"label": "C", "text": "Only the output we want", "correct": False},
                {"label": "D", "text": "A set of if-else statements", "correct": False},
            ]
        }
    },

    {
        "id": "b02_python_numpy",
        "level": "beginner",
        "order": 2,
        "title": "Python & NumPy Basics",
        "description": "Get comfortable with Python arrays and NumPy — the foundation of every ML pipeline.",
        "points": 2,
        "theory": """
<p>NumPy is the core numerical library in Python. Almost every ML framework (scikit-learn, TensorFlow, PyTorch)
uses NumPy arrays internally.</p>

<h5 class="mt-4">Key Concepts</h5>
<ul>
  <li><strong>ndarray</strong> — N-dimensional array, the fundamental data structure</li>
  <li><strong>Vectorised operations</strong> — apply operations to entire arrays at once, no loops needed</li>
  <li><strong>Broadcasting</strong> — operate on arrays of different shapes intelligently</li>
  <li><strong>Shape &amp; dtype</strong> — every array has a shape (dimensions) and dtype (data type)</li>
</ul>

<h5 class="mt-4">Creating Arrays</h5>
<pre><code>import numpy as np

a = np.array([1, 2, 3, 4, 5])   # 1D array
b = np.zeros((3, 4))             # 3x4 matrix of zeros
c = np.arange(0, 10, 2)         # [0, 2, 4, 6, 8]
d = np.linspace(0, 1, 5)        # [0.0, 0.25, 0.5, 0.75, 1.0]
</code></pre>

<h5 class="mt-4">Useful Operations</h5>
<pre><code>a.shape      # (5,)
a.mean()     # 3.0
a.std()      # standard deviation
a * 2        # [2, 4, 6, 8, 10] — no loop needed
a[a > 2]     # [3, 4, 5]  — boolean indexing
</code></pre>
""",
        "has_ide": True,
        "ide_language": "python",
        "ide_starter": """import numpy as np

# Create a 1D array of numbers 1 to 10
arr = np.arange(1, 11)

# TODO: Calculate and print the mean of the array
# TODO: Print only the elements greater than 5
""",
        "challenge_description": "Using NumPy, create an array of numbers 1–10, print its mean, and filter elements greater than 5. Your output must show the mean value and the filtered array.",
        "ide_solution_check": ["5.5", "6", "7", "8", "9", "10"],
        "question": {
            "text": "What does NumPy broadcasting allow you to do?",
            "options": [
                {"label": "A", "text": "Broadcast messages between arrays", "correct": False},
                {"label": "B", "text": "Operate on arrays of different shapes without explicit loops", "correct": True},
                {"label": "C", "text": "Convert arrays to strings automatically", "correct": False},
                {"label": "D", "text": "Duplicate array contents across memory", "correct": False},
            ]
        }
    },

    {
        "id": "b03_linear_regression",
        "level": "beginner",
        "order": 3,
        "title": "Linear Regression",
        "description": "Learn the simplest predictive model — fitting a line to data.",
        "points": 2,
        "theory": """
<p>Linear Regression models the relationship between a dependent variable <em>y</em> and one or more
independent variables <em>X</em> by fitting a straight line.</p>

<h5 class="mt-4">The Equation</h5>
<p class="formula-block"><code>y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ</code></p>
<p>Where <code>w₀</code> is the bias (intercept) and <code>w₁...wₙ</code> are the weights (coefficients).</p>

<h5 class="mt-4">How the Model Learns</h5>
<p>The model minimises the <strong>Mean Squared Error (MSE)</strong> — the average of squared differences
between predicted and actual values.</p>
<p class="formula-block"><code>MSE = (1/n) Σ (yᵢ - ŷᵢ)²</code></p>

<h5 class="mt-4">Using scikit-learn</h5>
<pre><code>from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, y)

print(model.coef_)       # slope
print(model.intercept_)  # intercept
print(model.predict([[6]]))  # predict for x=6
</code></pre>
""",
        "has_ide": True,
        "ide_language": "python",
        "ide_starter": """from sklearn.linear_model import LinearRegression
import numpy as np

# House size (sq ft) and price ($1000s)
X = np.array([[600], [800], [1000], [1200], [1500]])
y = np.array([150, 200, 250, 290, 360])

# TODO: Create and fit a LinearRegression model
# TODO: Predict the price of a 1300 sq ft house
# TODO: Print the prediction
""",
        "challenge_description": "Fit a linear regression model to predict house prices. Print the predicted price for a 1300 sq ft house. Your output should contain a number.",
        "ide_solution_check": ["predict", "270", "271", "272", "273", "274"],
        "question": {
            "text": "What does Linear Regression minimise during training?",
            "options": [
                {"label": "A", "text": "Cross-entropy loss", "correct": False},
                {"label": "B", "text": "Mean Squared Error (MSE)", "correct": True},
                {"label": "C", "text": "Mean Absolute Percentage Error", "correct": False},
                {"label": "D", "text": "KL Divergence", "correct": False},
            ]
        }
    },

    {
        "id": "b04_data_preprocessing",
        "level": "beginner",
        "order": 4,
        "title": "Data Preprocessing",
        "description": "Learn how to clean and prepare data before training any model.",
        "points": 2,
        "theory": """
<p>Real-world data is messy. Before training any model you need to handle missing values,
scale features, and encode categorical variables.</p>

<h5 class="mt-4">Common Preprocessing Steps</h5>
<ol>
  <li><strong>Handle missing values</strong> — drop rows, or fill with mean/median/mode</li>
  <li><strong>Feature scaling</strong> — standardise or normalise numeric features</li>
  <li><strong>Encode categoricals</strong> — convert strings to numbers (Label Encoding, One-Hot)</li>
  <li><strong>Train/test split</strong> — separate data for training and evaluation</li>
</ol>

<h5 class="mt-4">Standardisation (Z-score)</h5>
<p class="formula-block"><code>z = (x − μ) / σ</code></p>
<pre><code>from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
</code></pre>

<h5 class="mt-4">Train/Test Split</h5>
<pre><code>from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
</code></pre>
""",
        "has_ide": True,
        "ide_language": "python",
        "ide_starter": """import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample dataset
X = np.array([[1,200],[2,400],[3,300],[4,500],[5,150],[6,600],[7,250],[8,450],[9,350],[10,500]])
y = np.array([0,1,0,1,0,1,0,1,0,1])

# TODO: Split into 80% train, 20% test (random_state=42)
# TODO: Standardise X_train and X_test using StandardScaler
# TODO: Print the shape of X_train_scaled
""",
        "challenge_description": "Split the dataset and apply StandardScaler. Print the shape of the scaled training set. Output must contain '(8, 2)'.",
        "ide_solution_check": ["(8, 2)"],
        "question": {
            "text": "Why do we fit the StandardScaler on training data only and then transform both train and test?",
            "options": [
                {"label": "A", "text": "To make training faster", "correct": False},
                {"label": "B", "text": "To prevent data leakage from test set statistics", "correct": True},
                {"label": "C", "text": "Because test data has no mean or std", "correct": False},
                {"label": "D", "text": "The scaler only works on training data", "correct": False},
            ]
        }
    },

    # ═══════════════════════════ INTERMEDIATE ════════════════════════════════

    {
        "id": "i01_logistic_regression",
        "level": "intermediate",
        "order": 1,
        "title": "Logistic Regression",
        "description": "Extend linear regression to classification problems using the sigmoid function.",
        "points": 3,
        "theory": """
<p>Logistic Regression is used for <strong>binary classification</strong>. Instead of predicting a continuous
value, it predicts the probability that an input belongs to class 1.</p>

<h5 class="mt-4">The Sigmoid Function</h5>
<p class="formula-block"><code>σ(z) = 1 / (1 + e⁻ᶻ)</code></p>
<p>The sigmoid squashes any real number into the range (0, 1), making it interpretable as a probability.</p>

<h5 class="mt-4">Decision Boundary</h5>
<p>If <code>σ(z) ≥ 0.5</code>, predict class 1. Otherwise predict class 0.</p>

<h5 class="mt-4">Loss Function: Binary Cross-Entropy</h5>
<p class="formula-block"><code>L = −[y log(ŷ) + (1−y) log(1−ŷ)]</code></p>

<h5 class="mt-4">Implementation</h5>
<pre><code>from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
</code></pre>
""",
        "has_ide": True,
        "ide_language": "python",
        "ide_starter": """from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# TODO: Split data (test_size=0.2, random_state=42)
# TODO: Scale features
# TODO: Train LogisticRegression (max_iter=1000)
# TODO: Print accuracy on test set
""",
        "challenge_description": "Train a Logistic Regression on the breast cancer dataset and print the test accuracy. Output must contain 'accuracy' or a float value above 0.9.",
        "ide_solution_check": ["0.9", "accuracy", "Accuracy"],
        "question": {
            "text": "What is the output range of the sigmoid function?",
            "options": [
                {"label": "A", "text": "−∞ to +∞", "correct": False},
                {"label": "B", "text": "−1 to 1", "correct": False},
                {"label": "C", "text": "0 to 1", "correct": True},
                {"label": "D", "text": "0 to ∞", "correct": False},
            ]
        }
    },

    {
        "id": "i02_decision_trees",
        "level": "intermediate",
        "order": 2,
        "title": "Decision Trees & Random Forests",
        "description": "Tree-based models that partition data using feature thresholds.",
        "points": 3,
        "theory": """
<p>A <strong>Decision Tree</strong> recursively splits data based on feature values to make predictions.
<strong>Random Forests</strong> aggregate many trees to reduce variance and improve generalisation.</p>

<h5 class="mt-4">How a Tree Splits</h5>
<p>At each node, the algorithm finds the feature and threshold that best separates the classes,
measured by <strong>Gini impurity</strong> or <strong>Information Gain</strong>.</p>
<p class="formula-block"><code>Gini = 1 − Σ pᵢ²</code></p>

<h5 class="mt-4">Random Forest Key Ideas</h5>
<ul>
  <li><strong>Bagging</strong> — each tree trains on a random bootstrap sample of the data</li>
  <li><strong>Feature randomness</strong> — each split considers only a random subset of features</li>
  <li><strong>Majority vote</strong> — final prediction aggregates all trees</li>
</ul>

<h5 class="mt-4">Hyperparameters to Know</h5>
<ul>
  <li><code>max_depth</code> — limits tree depth to prevent overfitting</li>
  <li><code>n_estimators</code> — number of trees in the forest</li>
  <li><code>min_samples_split</code> — minimum samples needed to split a node</li>
</ul>
""",
        "has_ide": True,
        "ide_language": "python",
        "ide_starter": """from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Train a RandomForestClassifier with 100 trees (random_state=42)
# TODO: Print feature importances alongside feature names
# TODO: Print test accuracy
""",
        "challenge_description": "Train a Random Forest on the Iris dataset. Print test accuracy and feature importances. Output must contain 'accuracy' or a number above 0.9.",
        "ide_solution_check": ["0.9", "1.0", "accuracy", "importance"],
        "question": {
            "text": "What technique does Random Forest use to create diverse trees?",
            "options": [
                {"label": "A", "text": "Gradient boosting on residuals", "correct": False},
                {"label": "B", "text": "Bagging with random feature subsets", "correct": True},
                {"label": "C", "text": "Pruning each tree after training", "correct": False},
                {"label": "D", "text": "Sharing weights between trees", "correct": False},
            ]
        }
    },

    {
        "id": "i03_neural_networks",
        "level": "intermediate",
        "order": 3,
        "title": "Neural Networks",
        "description": "Build and understand feedforward neural networks from scratch conceptually.",
        "points": 3,
        "theory": """
<p>A Neural Network is a chain of linear transformations followed by non-linear <strong>activation functions</strong>,
stacked in layers.</p>

<h5 class="mt-4">Architecture</h5>
<ul>
  <li><strong>Input layer</strong> — receives raw features</li>
  <li><strong>Hidden layers</strong> — learn intermediate representations</li>
  <li><strong>Output layer</strong> — produces final prediction</li>
</ul>

<h5 class="mt-4">Common Activation Functions</h5>
<table class="table table-sm table-bordered">
  <thead><tr><th>Function</th><th>Formula</th><th>Use</th></tr></thead>
  <tbody>
    <tr><td>ReLU</td><td>max(0, x)</td><td>Hidden layers (default)</td></tr>
    <tr><td>Sigmoid</td><td>1/(1+e⁻ˣ)</td><td>Binary output</td></tr>
    <tr><td>Softmax</td><td>eˣⁱ / Σeˣʲ</td><td>Multi-class output</td></tr>
    <tr><td>Tanh</td><td>(eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)</td><td>Hidden layers (LSTM)</td></tr>
  </tbody>
</table>

<h5 class="mt-4">Training with Backpropagation</h5>
<p>The network computes a loss, then propagates gradients backward through the chain rule,
updating each weight by a small step proportional to the learning rate.</p>
""",
        "has_ide": True,
        "ide_language": "python",
        "ide_starter": """# We'll use scikit-learn's MLPClassifier (no GPU needed)
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_digits()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# TODO: Create MLPClassifier with hidden_layer_sizes=(128, 64), max_iter=300
# TODO: Fit and print test accuracy
""",
        "challenge_description": "Train a neural network on the digits dataset. Print test accuracy — should be above 0.95.",
        "ide_solution_check": ["0.9", "accuracy", "Accuracy"],
        "question": {
            "text": "What is the primary purpose of an activation function in a neural network?",
            "options": [
                {"label": "A", "text": "To speed up training by skipping layers", "correct": False},
                {"label": "B", "text": "To introduce non-linearity into the model", "correct": True},
                {"label": "C", "text": "To normalise inputs to the network", "correct": False},
                {"label": "D", "text": "To regularise and prevent overfitting", "correct": False},
            ]
        }
    },

    {
        "id": "i04_sql_for_ml",
        "level": "intermediate",
        "order": 4,
        "title": "SQL for ML Data Pipelines",
        "description": "Query and aggregate datasets using SQL — essential for real-world ML work.",
        "points": 3,
        "theory": """
<p>In production ML, data lives in databases. SQL is the universal language for extracting,
filtering, and aggregating that data before it reaches your model.</p>

<h5 class="mt-4">Core SQL for ML</h5>
<pre><code>-- Select features and label
SELECT age, income, education_level, churn
FROM customers
WHERE signup_date > '2022-01-01';

-- Aggregate statistics
SELECT
  category,
  AVG(purchase_value) AS avg_value,
  COUNT(*) AS n_samples
FROM transactions
GROUP BY category
HAVING COUNT(*) > 100;

-- Join feature tables
SELECT u.user_id, u.age, t.total_spent, l.label
FROM users u
JOIN transactions t ON u.user_id = t.user_id
JOIN labels l ON u.user_id = l.user_id;
</code></pre>

<h5 class="mt-4">Window Functions</h5>
<pre><code>-- Rolling average (useful for time-series features)
SELECT
  date,
  revenue,
  AVG(revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_avg
FROM daily_sales;
</code></pre>
""",
        "has_ide": True,
        "ide_language": "sql",
        "ide_starter": """-- We have a virtual 'employees' table with columns:
-- id, name, department, salary, years_experience

-- TODO: Write a query that returns:
-- The average salary per department
-- Only departments where average salary > 60000
-- Order by average salary descending

SELECT
  department,
  -- your query here
FROM employees
GROUP BY department
-- add HAVING and ORDER BY
""",
        "challenge_description": "Write a SQL query to find average salary per department, filtered to departments with avg salary > 60000, ordered descending. Your query must contain HAVING and ORDER BY.",
        "ide_solution_check": ["HAVING", "having", "ORDER BY", "order by", "AVG", "avg"],
        "question": {
            "text": "Which SQL clause filters groups after aggregation (like GROUP BY)?",
            "options": [
                {"label": "A", "text": "WHERE", "correct": False},
                {"label": "B", "text": "FILTER", "correct": False},
                {"label": "C", "text": "HAVING", "correct": True},
                {"label": "D", "text": "LIMIT", "correct": False},
            ]
        }
    },

    # ═══════════════════════════ ADVANCED ════════════════════════════════════

    {
        "id": "a01_cnn",
        "level": "advanced",
        "order": 1,
        "title": "Convolutional Neural Networks",
        "description": "Learn how CNNs extract spatial features from images using convolution and pooling.",
        "points": 5,
        "theory": """
<p>Convolutional Neural Networks (CNNs) are designed for data with spatial structure, primarily images.
They use <strong>convolution operations</strong> to detect local patterns regardless of position.</p>

<h5 class="mt-4">Key Layers</h5>
<ul>
  <li><strong>Conv2D</strong> — applies learnable filters that detect edges, textures, shapes</li>
  <li><strong>MaxPooling2D</strong> — downsamples feature maps, adding translation invariance</li>
  <li><strong>Flatten</strong> — converts 2D feature maps to 1D vector for Dense layers</li>
  <li><strong>Dropout</strong> — randomly zeroes activations during training to prevent overfitting</li>
</ul>

<h5 class="mt-4">Receptive Field</h5>
<p>Each neuron in a conv layer "sees" only a small region (the kernel) of the input.
Deeper layers aggregate larger regions, building up from edges → textures → objects.</p>

<h5 class="mt-4">Architecture Example (LeNet-style)</h5>
<pre><code>from tensorflow import keras

model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax'),
])
model.summary()
</code></pre>
""",
        "has_ide": True,
        "ide_language": "python",
        "ide_starter": """# Build a CNN architecture description using scikit-learn MLPClassifier
# (Keras/TF not available in sandbox - we'll simulate with MLP on flattened MNIST)
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

data = load_digits()
X = data.data / 16.0   # normalise pixel values
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Build a deep MLP (simulate CNN depth) with layers: (256, 128, 64)
# TODO: Fit and evaluate
# TODO: Print accuracy and the shape of the confusion matrix
""",
        "challenge_description": "Build a deep MLP simulating CNN depth on the digits dataset. Print accuracy and confusion matrix shape '(10, 10)'.",
        "ide_solution_check": ["(10, 10)", "accuracy", "0.9"],
        "question": {
            "text": "What is the main advantage of convolution over a fully connected layer for images?",
            "options": [
                {"label": "A", "text": "It trains faster on GPU", "correct": False},
                {"label": "B", "text": "It exploits spatial locality and parameter sharing", "correct": True},
                {"label": "C", "text": "It requires no activation function", "correct": False},
                {"label": "D", "text": "It works only on greyscale images", "correct": False},
            ]
        }
    },

    {
        "id": "a02_transformers",
        "level": "advanced",
        "order": 2,
        "title": "Transformers & Attention",
        "description": "Understand the architecture behind GPT, BERT, and modern LLMs.",
        "points": 5,
        "theory": """
<p>The Transformer architecture, introduced in "Attention Is All You Need" (2017), powers
virtually every modern large language model including GPT, BERT, and LLaMA.</p>

<h5 class="mt-4">Self-Attention</h5>
<p>For each token, self-attention computes how much to "attend" to every other token in the sequence.</p>
<p class="formula-block"><code>Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V</code></p>
<ul>
  <li><strong>Q (Query)</strong> — "what am I looking for?"</li>
  <li><strong>K (Key)</strong> — "what do I contain?"</li>
  <li><strong>V (Value)</strong> — "what do I actually pass forward?"</li>
</ul>

<h5 class="mt-4">Multi-Head Attention</h5>
<p>Run multiple attention heads in parallel, each learning different relationship types
(syntax, semantics, coreference, etc.), then concatenate results.</p>

<h5 class="mt-4">Positional Encoding</h5>
<p>Transformers have no recurrence, so position is injected via sinusoidal encodings added to embeddings.</p>

<h5 class="mt-4">Why Transformers Won</h5>
<ul>
  <li>Parallelisable — no sequential dependency unlike RNNs</li>
  <li>Long-range dependencies — any two tokens directly attend to each other</li>
  <li>Scalable — performance keeps improving with more data and parameters</li>
</ul>
""",
        "has_ide": True,
        "ide_language": "python",
        "ide_starter": """import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    # TODO: Implement the attention formula
    # Attention(Q,K,V) = softmax(Q @ K.T / sqrt(d_k)) @ V
    d_k = Q.shape[-1]
    # your code here
    pass

# Test your implementation
np.random.seed(42)
Q = np.random.randn(3, 4)   # 3 tokens, dim 4
K = np.random.randn(3, 4)
V = np.random.randn(3, 4)

result = scaled_dot_product_attention(Q, K, V)
print("Output shape:", result.shape)   # should be (3, 4)
print("Output:\\n", result)
""",
        "challenge_description": "Implement scaled dot-product attention from scratch. Output shape must be (3, 4).",
        "ide_solution_check": ["(3, 4)", "Output shape"],
        "question": {
            "text": "Why is the dot product divided by √dₖ in the attention formula?",
            "options": [
                {"label": "A", "text": "To make the output sum to 1", "correct": False},
                {"label": "B", "text": "To prevent vanishing gradients in early layers", "correct": False},
                {"label": "C", "text": "To prevent large dot products from pushing softmax into saturation", "correct": True},
                {"label": "D", "text": "To normalise the query and key vectors", "correct": False},
            ]
        }
    },

    {
        "id": "a03_js_data_viz",
        "level": "advanced",
        "order": 3,
        "title": "Data Visualisation with JavaScript",
        "description": "Build interactive ML visualisations using vanilla JS and Canvas.",
        "points": 5,
        "theory": """
<p>Understanding and communicating ML results often requires interactive visualisation.
JavaScript runs in the browser making it ideal for shareable, interactive charts.</p>

<h5 class="mt-4">Canvas API Basics</h5>
<pre><code>const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');

// Draw a point
ctx.beginPath();
ctx.arc(x, y, radius, 0, Math.PI * 2);
ctx.fillStyle = 'steelblue';
ctx.fill();
</code></pre>

<h5 class="mt-4">Visualising a Decision Boundary</h5>
<p>For a 2D classifier, shade each pixel by its predicted class to show the boundary the model learned.</p>

<h5 class="mt-4">Why This Matters in ML</h5>
<ul>
  <li>Spot class imbalance immediately from a scatter plot</li>
  <li>See if clusters are linearly separable</li>
  <li>Diagnose overfitting from training vs validation loss curves</li>
  <li>Explain model behaviour to non-technical stakeholders</li>
</ul>
""",
        "has_ide": True,
        "ide_language": "javascript",
        "ide_starter": """// Visualise a simple 2D dataset as a scatter plot
// Using console output to represent the "chart" in text form

const data = [
  {x: 1.2, y: 2.3, label: 'A'},
  {x: 2.1, y: 3.1, label: 'A'},
  {x: 5.4, y: 6.2, label: 'B'},
  {x: 6.1, y: 5.8, label: 'B'},
  {x: 3.3, y: 4.1, label: 'A'},
  {x: 7.2, y: 8.1, label: 'B'},
];

// TODO: Group data by label and print count per label
// TODO: Calculate centroid (mean x, mean y) for each label
// TODO: Print a simple ASCII representation showing label counts
""",
        "challenge_description": "Group the dataset by label, compute centroids, and print them. Output must contain 'centroid' or 'A' and 'B' labels.",
        "ide_solution_check": ["centroid", "Centroid", "label A", "label B", "Label A", "Label B", ": A", ": B"],
        "question": {
            "text": "What does a decision boundary visualisation show?",
            "options": [
                {"label": "A", "text": "The training loss over time", "correct": False},
                {"label": "B", "text": "The regions of input space assigned to each class by the model", "correct": True},
                {"label": "C", "text": "The distribution of feature values", "correct": False},
                {"label": "D", "text": "The confusion matrix as a heatmap", "correct": False},
            ]
        }
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

LEVEL_ORDER = ["beginner", "intermediate", "advanced"]

LEVEL_META = {
    "beginner": {
        "label": "Beginner",
        "description": "Core concepts, Python, NumPy, and your first models.",
        "icon": "🌱",
        "color": "#27a35a",
        "unlock_points": 0,
        "quiz_pass_score": 0,   # always unlocked
    },
    "intermediate": {
        "label": "Intermediate",
        "description": "Classification, tree models, neural networks, and SQL.",
        "icon": "⚡",
        "color": "#b84c27",
        "unlock_points": 5,
        "quiz_pass_score": 3,   # 3/5 correct MCQ + pass code challenge
    },
    "advanced": {
        "label": "Advanced",
        "description": "CNNs, Transformers, and production-grade techniques.",
        "icon": "🔬",
        "color": "#1a2535",
        "unlock_points": 8,
        "quiz_pass_score": 3,
    },
}

# Unlock quiz questions per level (5 MCQ + 1 code challenge each)
UNLOCK_QUIZZES: Dict[str, Dict] = {
    "intermediate": {
        "mcq": [
            {
                "id": "uq_i_1",
                "text": "Which loss function is used in Logistic Regression?",
                "options": [
                    {"label": "A", "text": "Mean Squared Error", "correct": False},
                    {"label": "B", "text": "Binary Cross-Entropy", "correct": True},
                    {"label": "C", "text": "Hinge Loss", "correct": False},
                    {"label": "D", "text": "Huber Loss", "correct": False},
                ]
            },
            {
                "id": "uq_i_2",
                "text": "What does StandardScaler do to each feature?",
                "options": [
                    {"label": "A", "text": "Scales to range [0, 1]", "correct": False},
                    {"label": "B", "text": "Removes mean and scales to unit variance", "correct": True},
                    {"label": "C", "text": "Converts to log scale", "correct": False},
                    {"label": "D", "text": "Rounds to nearest integer", "correct": False},
                ]
            },
            {
                "id": "uq_i_3",
                "text": "What is the purpose of a train/test split?",
                "options": [
                    {"label": "A", "text": "To speed up training", "correct": False},
                    {"label": "B", "text": "To evaluate model performance on unseen data", "correct": True},
                    {"label": "C", "text": "To reduce the dataset size", "correct": False},
                    {"label": "D", "text": "To balance class distribution", "correct": False},
                ]
            },
            {
                "id": "uq_i_4",
                "text": "Which NumPy function creates an array of evenly spaced values?",
                "options": [
                    {"label": "A", "text": "np.zeros()", "correct": False},
                    {"label": "B", "text": "np.random.randn()", "correct": False},
                    {"label": "C", "text": "np.linspace()", "correct": True},
                    {"label": "D", "text": "np.reshape()", "correct": False},
                ]
            },
            {
                "id": "uq_i_5",
                "text": "Linear Regression predicts what type of output?",
                "options": [
                    {"label": "A", "text": "A class label", "correct": False},
                    {"label": "B", "text": "A probability between 0 and 1", "correct": False},
                    {"label": "C", "text": "A continuous numeric value", "correct": True},
                    {"label": "D", "text": "A cluster assignment", "correct": False},
                ]
            },
        ],
        "code": {
            "description": "Write a Python function called `mse` that takes two lists `y_true` and `y_pred` and returns the Mean Squared Error. Call it with y_true=[1,2,3] and y_pred=[1.1,2.2,2.9] and print the result.",
            "language": "python",
            "starter": """def mse(y_true, y_pred):
    # TODO: implement MSE = mean of squared differences
    pass

# Test
result = mse([1, 2, 3], [1.1, 2.2, 2.9])
print(f"MSE: {result:.4f}")
""",
            "solution_check": ["MSE:", "0.02", "0.0", "mse"],
        }
    },
    "advanced": {
        "mcq": [
            {
                "id": "uq_a_1",
                "text": "What is the key advantage of Random Forest over a single Decision Tree?",
                "options": [
                    {"label": "A", "text": "It trains on more data", "correct": False},
                    {"label": "B", "text": "It reduces variance through ensemble averaging", "correct": True},
                    {"label": "C", "text": "It has fewer hyperparameters", "correct": False},
                    {"label": "D", "text": "It always achieves 100% training accuracy", "correct": False},
                ]
            },
            {
                "id": "uq_a_2",
                "text": "Which activation function is the default choice for hidden layers in modern networks?",
                "options": [
                    {"label": "A", "text": "Sigmoid", "correct": False},
                    {"label": "B", "text": "Tanh", "correct": False},
                    {"label": "C", "text": "ReLU", "correct": True},
                    {"label": "D", "text": "Softmax", "correct": False},
                ]
            },
            {
                "id": "uq_a_3",
                "text": "What does the HAVING clause do in SQL?",
                "options": [
                    {"label": "A", "text": "Filters rows before grouping", "correct": False},
                    {"label": "B", "text": "Filters groups after aggregation", "correct": True},
                    {"label": "C", "text": "Joins two tables", "correct": False},
                    {"label": "D", "text": "Orders the results", "correct": False},
                ]
            },
            {
                "id": "uq_a_4",
                "text": "Backpropagation updates weights using:",
                "options": [
                    {"label": "A", "text": "Random search", "correct": False},
                    {"label": "B", "text": "Genetic algorithms", "correct": False},
                    {"label": "C", "text": "Gradient of the loss w.r.t. each weight", "correct": True},
                    {"label": "D", "text": "Second-order Newton's method only", "correct": False},
                ]
            },
            {
                "id": "uq_a_5",
                "text": "What does Dropout do during training?",
                "options": [
                    {"label": "A", "text": "Removes low-importance features permanently", "correct": False},
                    {"label": "B", "text": "Randomly zeroes neuron activations to prevent co-adaptation", "correct": True},
                    {"label": "C", "text": "Reduces the learning rate", "correct": False},
                    {"label": "D", "text": "Clips gradient values", "correct": False},
                ]
            },
        ],
        "code": {
            "description": "Implement a sigmoid function and its derivative in Python. Print sigmoid(0), sigmoid(2), and sigmoid_derivative(0). sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x)).",
            "language": "python",
            "starter": """import math

def sigmoid(x):
    # TODO: implement
    pass

def sigmoid_derivative(x):
    # TODO: implement using sigmoid
    pass

print(f"sigmoid(0) = {sigmoid(0):.4f}")
print(f"sigmoid(2) = {sigmoid(2):.4f}")
print(f"sigmoid_derivative(0) = {sigmoid_derivative(0):.4f}")
""",
            "solution_check": ["0.5000", "0.8808", "0.2500", "sigmoid(0)"],
        }
    }
}


def get_modules_by_level(level: str) -> List[Dict]:
    return sorted(
        [m for m in MODULES if m["level"] == level],
        key=lambda m: m["order"]
    )


def get_module_by_id(module_id: str) -> Dict | None:
    return next((m for m in MODULES if m["id"] == module_id), None)


def get_unlock_quiz(level: str) -> Dict | None:
    return UNLOCK_QUIZZES.get(level)
