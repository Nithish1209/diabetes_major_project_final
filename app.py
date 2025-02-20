#working - prediction history saved
from flask import Flask, render_template, request, jsonify
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import numpy as np
import os

# Global variables
X_train, X_test, y_train, y_test = None, None, None, None
dataset = None
decision, decision_acc = None, 0
svm, svm_acc = None, 0
ann, ann_acc = None, 0
random_forest, random_forest_acc = None, 0
ensemble, ensemble_acc = None, 0

app = Flask(__name__)

def preprocess():
    global X_train, y_train, dataset, X_test, y_test
    pima = fetch_openml(name='diabetes', version=1, as_frame=True)
    dataset = pima.frame

    dataset.columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    dataset['Outcome'] = dataset['Outcome'].map({'tested_positive': 1, 'tested_negative': 0})

    #sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    #plt.savefig('static/heatmap.png')

    dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
    #msno.bar(dataset)
    #plt.savefig('static/missing_values.png')

    for col in dataset.columns[:-1]:  
        median_values = dataset.groupby('Outcome')[col].median()
        dataset.loc[dataset['Outcome'] == 0, col] = dataset.loc[dataset['Outcome'] == 0, col].fillna(median_values[0])
        dataset.loc[dataset['Outcome'] == 1, col] = dataset.loc[dataset['Outcome'] == 1, col].fillna(median_values[1])

    for col in dataset.columns[:-1]:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        dataset[col] = np.where(dataset[col] > upper, upper, dataset[col])
        dataset[col] = np.where(dataset[col] < lower, lower, dataset[col])
    
    lof = LocalOutlierFactor(n_neighbors=10)
    outlier_predictions = lof.fit_predict(dataset.drop('Outcome', axis=1))
    dataset = dataset[outlier_predictions == 1]

    dataset.to_csv("preprocessed_diabetes.csv", index=False)

    X = dataset.drop('Outcome', axis=1)
    y = dataset['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

def calculateAllAccuracies():
    global decision, decision_acc, svm, svm_acc, ann, ann_acc, random_forest, random_forest_acc, ensemble, ensemble_acc

    if X_train is None or y_train is None:
        return {"error": "Data is not preprocessed yet. Please preprocess the data first."}

    decision = DecisionTreeClassifier(max_depth=5)
    decision.fit(X_train, y_train)
    decision_acc = accuracy_score(y_test, decision.predict(X_test)) * 100

    svm = SVC(C=1.0, gamma='scale', kernel='linear', random_state=2)
    svm.fit(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test)) * 100

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ann = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000, learning_rate_init=0.0001)
    ann.fit(X_train_scaled, y_train)
    ann_acc = accuracy_score(y_test, ann.predict(X_test_scaled)) * 100

    random_forest = RandomForestClassifier(n_estimators=100, random_state=1)
    random_forest.fit(X_train, y_train)
    random_forest_acc = accuracy_score(y_test, random_forest.predict(X_test)) * 100

    estimators = [('tree', decision), ('svm', svm), ('ann', ann), ('rf', random_forest)]
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(X_train, y_train)
    ensemble_acc = accuracy_score(y_test, ensemble.predict(X_test)) * 100

    return {
        "decision_acc": decision_acc,
        "svm_acc": svm_acc,
        "ann_acc": ann_acc,
        "random_forest_acc": random_forest_acc,
        "ensemble_acc": ensemble_acc
    }

def predictOutcome(input_values):
    global ensemble
    if ensemble is None:
        return {"error": "Please train the Ensemble model first!"}

    try:
        input_df = pd.DataFrame([input_values], columns=X_train.columns)
        prediction = ensemble.predict(input_df)
        outcome = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        history_file = "prediction_history.csv"
        record = input_values + [outcome]
        columns = X_train.columns.tolist() + ["Prediction"]

        if not os.path.exists(history_file):
            pd.DataFrame([record], columns=columns).to_csv(history_file, index=False)
        else:
            pd.DataFrame([record], columns=columns).to_csv(history_file, mode='a', header=False, index=False)

        return {"outcome": outcome}
    except ValueError:
        return {"error": "Invalid input values."}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess_route():
    preprocess()
    return jsonify({"message": "Data Preprocessed successfully."})

@app.route('/calculate_accuracies', methods=['POST'])
def accuracies_route():
    accuracies = calculateAllAccuracies()
    return jsonify(accuracies)

@app.route('/predict', methods=['POST'])
def predict_route():
    input_values = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['blood_pressure']),
        float(request.form['skin_thickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    ]
    prediction = predictOutcome(input_values)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)



'''
#working
from flask import Flask, render_template, request, jsonify
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import numpy as np

# Global variables
X_train, X_test, y_train, y_test = None, None, None, None
dataset = None
decision, decision_acc = None, 0
svm, svm_acc = None, 0
ann, ann_acc = None, 0
random_forest, random_forest_acc = None, 0
ensemble, ensemble_acc = None, 0

app = Flask(__name__)

# Function to preprocess the dataset and split into X and y
def preprocess():
    global X_train, y_train, dataset, X_test, y_test
    pima = fetch_openml(name='diabetes', version=1, as_frame=True)
    dataset = pima.frame

    # Rename columns
    dataset.columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    # Map Outcome values
    dataset['Outcome'] = dataset['Outcome'].map({'tested_positive': 1, 'tested_negative': 0})

    # EDA
    sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.savefig('static/heatmap.png')

    # Handle Missing Values
    dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = dataset[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
    msno.bar(dataset)
    plt.savefig('static/missing_values.png')

    # Fill missing values
    for col in dataset.columns[:-1]:  
        median_values = dataset.groupby('Outcome')[col].median()
        dataset.loc[dataset['Outcome'] == 0, col] = dataset.loc[dataset['Outcome'] == 0, col].fillna(median_values[0])
        dataset.loc[dataset['Outcome'] == 1, col] = dataset.loc[dataset['Outcome'] == 1, col].fillna(median_values[1])

    # Outlier Detection
    for col in dataset.columns[:-1]:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        dataset[col] = np.where(dataset[col] > upper, upper, dataset[col])
        dataset[col] = np.where(dataset[col] < lower, lower, dataset[col])
    
    lof = LocalOutlierFactor(n_neighbors=10)
    outlier_predictions = lof.fit_predict(dataset.drop('Outcome', axis=1))
    dataset = dataset[outlier_predictions == 1]

    # Save preprocessed dataset
    dataset.to_csv("preprocessed_diabetes.csv", index=False)

    # Splitting data
    X = dataset.drop('Outcome', axis=1)
    y = dataset['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Function to calculate all model accuracies
def calculateAllAccuracies():
    global decision, decision_acc, svm, svm_acc, ann, ann_acc, random_forest, random_forest_acc, ensemble, ensemble_acc

    if X_train is None or y_train is None:
        return {"error": "Data is not preprocessed yet. Please preprocess the data first."}

    # Decision Tree
    decision = DecisionTreeClassifier(max_depth=5)
    decision.fit(X_train, y_train)
    y_pred = decision.predict(X_test)
    decision_acc = accuracy_score(y_test, y_pred) * 100

    # SVM
    svm = SVC(C=1.0, gamma='scale', kernel='linear', random_state=2)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred) * 100

    # ANN
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ann = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000, learning_rate_init=0.0001)
    ann.fit(X_train_scaled, y_train)
    y_pred = ann.predict(X_test_scaled)
    ann_acc = accuracy_score(y_test, y_pred) * 100

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=100, random_state=1)
    random_forest.fit(X_train, y_train)
    y_pred = random_forest.predict(X_test)
    random_forest_acc = accuracy_score(y_test, y_pred) * 100

    # Ensemble Model
    estimators = [
        ('tree', decision),
        ('svm', svm),
        ('ann', ann),
        ('rf', random_forest)
    ]
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, y_pred) * 100

    return {
        "decision_acc": decision_acc,
        "svm_acc": svm_acc,
        "ann_acc": ann_acc,
        "random_forest_acc": random_forest_acc,
        "ensemble_acc": ensemble_acc
    }

# Prediction function
def predictOutcome(input_values):
    global ensemble
    if ensemble is None:
        return {"error": "Please train the Ensemble model first!"}

    try:
        input_df = pd.DataFrame([input_values], columns=X_train.columns)
        prediction = ensemble.predict(input_df)
        outcome = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        history_file = "prediction_history.csv"
        record = input_values + [outcome]
        columns = X_train.columns.tolist() + ["Prediction"]

        if not os.path.exists(history_file):
            pd.DataFrame([record], columns=columns).to_csv(history_file, index=False)
        else:
            pd.DataFrame([record], columns=columns).to_csv(history_file, mode='a', header=False, index=False)

        return {"outcome": outcome}
    except ValueError:
        return {"error": "Invalid input values."}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preprocess', methods=['POST'])
def preprocess_route():
    preprocess()
    return jsonify({"message": "Data Preprocessed successfully."})

@app.route('/calculate_accuracies', methods=['POST'])
def accuracies_route():
    accuracies = calculateAllAccuracies()
    return jsonify(accuracies)

@app.route('/predict', methods=['POST'])
def predict_route():
    input_values = [
        float(request.form['pregnancies']),
        float(request.form['glucose']),
        float(request.form['blood_pressure']),
        float(request.form['skin_thickness']),
        float(request.form['insulin']),
        float(request.form['bmi']),
        float(request.form['dpf']),
        float(request.form['age'])
    ]
    prediction = predictOutcome(input_values)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
'''
