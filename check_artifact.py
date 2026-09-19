#recharger le modèle depuis le disque pour pouvoir l'utiliser pour faire des prédictions sur de nouvelles données.

#predire une seule ligne de données, affiche la proba avec la version du modèle.
import joblib
import pandas as pd
import json

#fonction qui prends en params pipeline et les varriables
def predict_single_row(pipeline, row):
    """
    Predict the class for a single row of data using the trained pipeline.
    Args:
        pipeline (Pipeline): The trained pipeline object.
        row (pd.DataFrame): A single row of data to predict.
    Returns:
        float: The predicted probability of the positive class.
    """
    # Convert the row to a DataFrame

    # Make prediction
    y_proba = pipeline.predict_proba(row)

    # Return the probability of the positive class (1)
    return y_proba[0][1]


def main():
    # Load the trained model
    pipeline = joblib.load("artifacts/model.joblib")
    with open("artifacts/model_meta.json", "r") as f:
        json_data = json.load(f)

    load_data = pd.read_csv("data/raw.csv") 
    variables = json_data["variables"]

    variables_used = load_data[variables]

    first_row = variables_used.iloc[0]

    first_row_df = pd.DataFrame([first_row], columns=variables_used.columns)

    predicted_proba = predict_single_row(pipeline, first_row_df)
    print(f"Predicted probability for the first row: {predicted_proba}")

main()