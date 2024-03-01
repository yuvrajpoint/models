from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Loading CNN model
cnn_model = load_model('cnn_ml_model.h5')

# Defining a route for the home page
@app.route('/')
def home():
    return render_template('index.html', predictions=None, predictions2=None, accuracy=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file from the request
        uploaded_file = request.files['file']

        # Read the uploaded file into a Pandas DataFrame
        input_data = pd.read_csv(uploaded_file, sep='|')

        # Extract features
        x_new = input_data.drop(columns=['legitimate', 'Name', 'md5'], axis=1)
        y_new = input_data['legitimate']

        # Preprocess the data for the CNN model
        X_reshaped = np.expand_dims(x_new.values, axis=-1)

        # Make predictions using the model
        predictions = (cnn_model.predict(X_reshaped) > 0.5).astype(int)

        # Calculating the accuracy 
        test_loss, test_accuracy = cnn_model.evaluate(X_reshaped, y_new)
        legit_count = 0
        mal_count = 0
        for i in predictions:
            if (i == 1):
                legit_count += 1
            elif (i == 0):
                mal_count += 1
        res1 = "Total no. of legit files = " + str(legit_count)
        res2 = " Total no. of malware files = " + str(mal_count)
        acc = "Accuracy = " + str(test_accuracy)
        
        # Render the HTML template with predictions
        return render_template('index.html', predictions=res1, predictions2=res2, accuracy=acc)
'''
if __name__ == '__main__':
    app.run(debug=True)
'''
