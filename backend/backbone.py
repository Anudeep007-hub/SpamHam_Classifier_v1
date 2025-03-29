from CleanText import SentTokenize,Token_to_Sent 
from  RemoveStopWords import stopWords_removal
from PorterStemmer import porter_stemmer
import pickle
import json
import numpy as np  




# Load the stored model and vectorizer
with open("model_results/svm_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

with open("model_results/vectorizer.pkl", "rb") as f:
    loaded_vectorizer = pickle.load(f)


def get_predicted_class(sentence):
    """
    Predicts the class of a given sentence using the trained SVM model.

    Args:
        sentence (str): Input sentence to classify.

    Returns:
        int: Predicted class (0 or 1).
    """
    # Transform the input sentence using the stored vectorizer
    transformed_input = loaded_vectorizer.transform([sentence]).toarray()

    # Predict class
    prediction = loaded_model.predict(transformed_input)
    
    print(f"Predicted Class: {prediction[0]}")
    return prediction[0]
