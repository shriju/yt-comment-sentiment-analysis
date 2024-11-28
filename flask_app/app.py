from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow
import numpy as np
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient  # Corrected import

app = Flask(__name__)

CORS(app)

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # convert to lower case
        comment = comment.lower()
        # remove trailing and leading whitespaces
        comment = comment.strip()
        # remove newline characters
        comment = re.sub(r'\n', '', comment)
        # remove punctuation
        comment = re.sub(r'[^\w\s]', '', comment)
        # remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        # lemmatize words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    """Load the model and vectorizer from the model registry and local storage."""
    # Set MLflow tracking URI to the AWS server
    mlflow.set_tracking_uri('http://ec2-3-15-38-250.us-east-2.compute.amazonaws.com:5000')

    # Get the model version from the registry
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"  # Ensure there is no space after `models:/`
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", "./tfidf_vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # Make predictions
        predictions = model.predict(transformed_comments).tolist()

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Return the response with original comments and predicted sentiments
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
