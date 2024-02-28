from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load('malware_detection_model_new.joblib')

@app.route('/')
def home():
    return render_template('index.html', predictions=None, predictions2=None, accuracy=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file from the request
        uploaded_file = request.files['file']

        # Reading the file
        input_data = pd.read_csv(uploaded_file, sep='|')

        # Make predictions using the model
        x_new = input_data.drop(columns=['legitimate', 'Name', 'md5'], axis=1)
        y_new = input_data['legitimate']
        predictions = model.predict(x_new)
        accuracy = model.score(x_new, y_new)
        legit_count = 0;
        mal_count = 0;
        for i in predictions:
            if (i == 1):
                legit_count += 1
            elif (i == 0):
                mal_count += 1
        res1 = "Total no. of legit files = " + str(legit_count)
        res2 = " Total no. of malware files = " + str(mal_count)
        acc = "Accuracy = " + str(accuracy)
        
        # Render the HTML template with predictions
        return render_template('index.html', predictions=res1, predictions2=res2, accuracy=acc)
'''
if __name__ == '__main__':
    app.run(debug=True)
'''
