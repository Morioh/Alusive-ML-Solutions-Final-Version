import os
import tempfile
import numpy as np
from PIL import Image
from pdf2image import convert_from_path
import tensorflow as tf
import resend
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from decouple import config

# Load the trained model for grant prediction
GRANT_MODEL_PATH = "models/rf_grant_model.pkl"
rf_clf = joblib.load(GRANT_MODEL_PATH)

# Load the trained model for document validation
DOCUMENT_MODEL_PATH = "models/document_validator.h5"


class DocumentValidator:
    def __init__(self, model_path=DOCUMENT_MODEL_PATH):
        self.model = tf.keras.models.load_model(model_path)
        self.classes = ["unsigned", "signed"]

    def validate_document(self, file_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            if file_path.lower().endswith(".pdf"):
                images = convert_from_path(file_path, output_folder=temp_dir)
                image_paths = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir)]
            else:
                image_paths = [file_path]

            last_page_path = image_paths[-1]
            processed_img = preprocess_image(last_page_path)
            pred = self.model.predict(processed_img, verbose=0)[0][0]

        is_signed = pred > 0.75
        return {
            "prediction": self.classes[int(is_signed)],
            "signed_probability": float(pred),
            "last_page_analysis": {
                "page": len(image_paths),
                "signed": bool(is_signed),
                "confidence": float(pred),
            },
        }


def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocesses image for document validation model."""
    img = Image.open(img_path)
    img = img.convert("RGB").resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


# Initialize document validator
validator = DocumentValidator()


def validate_document(file_path):
    """Validate the document and return the analysis result."""
    return validator.validate_document(file_path)


# Send emails using Resend API
resend.api_key = config("RESEND_API_KEY")


def send_email(to_email, body):
    """Send an email using the Resend API."""
    params = {
        # Use "from" instead of "from_"
        "from": "Alusive <alusiveafrica_rwa@alusiveafrica.org>",
        "to": [to_email, 'alusiveafrica_rwa@alustudent.com'],
        "subject": "Status of your Uploaded Document",
        "html": body,
    }

    email = resend.Emails.send(params)
    return email


# Alusive Grant Giver
GRANT_MESSAGES = {
    0: "You qualify for up to $400 of funding pending the verification of supporting documents by ALU Financial Aid.",
    1: "You qualify for up to $700 of funding pending the verification of supporting documents by ALU Financial Aid.",
    2: "You qualify for up to $1000 of funding pending the verification of supporting documents by ALU Financial Aid."
}


def compute_features(df):
    """Computes additional required features before model prediction."""
    df["dependants_per_supporter"] = df["Household Dependants"] / (df["Household Supporters"] + 1)  # Avoid div by zero
    df["fee_to_income"] = df["Fee balance (USD)"] / (df["Total Monthly Income"] + 1)  # Avoid div by zero
    df["household_income_per_person"] = df["Total Monthly Income"] / df["Household Size"]
    df["requested_to_affordable"] = df["Grant Requested"] / (df["Amount Affordable"] + 1)  # Avoid div by zero
    return df


def preprocess_input(data: dict, feature_columns: list):
    """
    Preprocess input data into a DataFrame, ensuring one-hot encoding and feature alignment.
    """
    df = pd.DataFrame([data])

    # Compute missing features
    df = compute_features(df)

    # One-hot encode categorical features
    categorical_features = ["Academic Standing", "Disciplinary Standing", "Financial Standing",
                            "ALU Grant Status", "Previous Alusive Grant Status"]
    df_encoded = pd.get_dummies(df, columns=categorical_features)

    # Add missing columns that were in training data
    missing_cols = set(feature_columns) - set(df_encoded.columns)
    for col in missing_cols:
        df_encoded[col] = 0

    # Ensure correct column order
    df_encoded = df_encoded[feature_columns]
    
    return df_encoded


def predict_grant_category(applicant_data: dict, feature_columns: list):
    """
    Predicts grant category based on applicant data.
    Returns predicted category, class probabilities, and grant message.
    """
    processed_data = preprocess_input(applicant_data, feature_columns)

    # Model prediction
    predicted_category = rf_clf.predict(processed_data)[0]
    predicted_probabilities = rf_clf.predict_proba(processed_data)[0]

    return {
        "predicted_category": int(predicted_category),
        "probabilities": predicted_probabilities.tolist(),
        "grant_message": GRANT_MESSAGES.get(predicted_category, "Error: Invalid category predicted.")
    }
