from flask import Flask, render_template, request
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Load the Isolation Forest model
model_path = "isolation_forest.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        data = pd.read_csv(f)

        X = data.drop("Class", axis=1)
        y = data["Class"]

        y_pred = model.predict(X)
        y_pred[y_pred==1]=0
        y_pred[y_pred==-1]=1

        conf_matrix = confusion_matrix(y, y_pred)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)

        errors_original = y != y_pred
        mislabel_df = pd.DataFrame(data=X[errors_original], columns=X.columns)
        mislabel_df['Predicted_Label'] = y_pred[errors_original]
        mislabel_df['True_Label'] = y[errors_original]

        plt.figure(figsize=(6, 6)) 
        sns.heatmap(pd.DataFrame(conf_matrix), xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'], 
                    linewidths=0.05, annot=True, fmt="d", cmap='BuPu')
        plt.title("Isolation Forest Classifier - Confusion Matrix")
        plt.xlabel('Predicted Value')
        plt.ylabel('True Value')
        plt.savefig('static/confusion_matrix.png')

        return render_template('result.html', accuracy=accuracy, report=report, mislabel_df=mislabel_df.to_html(), 
                                image='static/confusion_matrix.png')

if __name__ == '__main__':
    app.run(debug=True)
