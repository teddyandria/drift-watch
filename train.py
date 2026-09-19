import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from pathlib import Path
import json
from datetime import datetime

#   etape 1 : charger le csv
def load_data(file_path) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded dataset as a DataFrame."""
    df = pd.read_csv(file_path)
    return df

def separate_target(df) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the features and target variable from the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        tuple: A tuple containing the features (X) and target (y) DataFrames."""
    X = df.drop(columns=["class"])
    y = df["class"]

    #conversion binaire. (pas de boucle lorsqu'on manipule pandas, on peut utiliser la méthode map pour transformer les valeurs de la colonne "class" en 0 et 1)
    y = y.map({"bad": 1, "good": 0})
    #taux de défaut moyen
    # print(y.mean()) 
    return X, y


#étape 2 : split du dataset en train et test (80/20). stratifiy permet de conserver la même proportion de classes dans les deux datasets. random_state permet de reproduire le split.

def split_data(X, y) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test



#etape 3 : Prétraitement des données
#ColumnTransformer permet de transformer et traiter les variables afin d'avoir un dataset prêt pour l'entrainement du modèle.
def preprocess_data(X_train) -> ColumnTransformer:
    """Preprocess the data using ColumnTransformer.
    Args:
        X_train (pd.DataFrame): The training data.
    Returns:
        ColumnTransformer: The preprocessor object.
    """
    numerical_features = X_train.select_dtypes(include=["number"]).columns
    categorical_features = X_train.select_dtypes(include=["str"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features) 
        ]
    )

    return preprocessor

def train_model(X_train, y_train, X_test, y_test, preprocessor) -> Pipeline:
    """Train a logistic regression model using a pipepline.
    Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.
        preprocessor (ColumnTransformer): The preprocessor object.
    Returns:
        Pipeline: The trained pipeline object.
    """
    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    print(pipeline.score(X_test, y_test))
    print("Model training completed.")

    #calcul AUC sur jeu de test
    y_proba = pipeline.predict_proba(X_test)

    #[:, 1] car on veut la probabilité de la classe positive (1)
    auc = roc_auc_score(y_test, y_proba[:, 1])

    #calcul gini sur jeu de test
    gini = 2 * auc - 1

    print(f"AUC: {auc}")
    print(f"Gini: {gini}")
    result = {
        "AUC": auc,
        "Gini": gini,
        "pipeline": pipeline
    }
    return result

#joblib permet de sauvegarder le modèle entrainé dans un fichier .joblib pour pouvoir le réutiliser plus tard sans avoir à le réentrainer.
#sauvegarde du modèle entrainé dans un fichier .joblib

def save_model(pipeline, X_test, y_test)-> None:
    """
    Save the trained model to a file using joblib.
    Args:
        pipeline (Pipeline): The trained pipeline object.
        X_test (pd.DataFrame): The testing features.
        y_test (pd.Series): The testing target.
    Returns:
        None
    """
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, "artifacts/model.joblib")

    #utiliser joblib.load pour charger le modèle et faire des prédictions sur de nouvelles données.
    model_loaded = joblib.load("artifacts/model.joblib")
    score =model_loaded.score(X_test, y_test)

    print(score)

    return None

def save_mertrics(auc, variables):
    """
    Save the model metrics to a JSON file.
    Args:
        auc (float): The AUC score of the model.
        variables (list): The list of feature names used in the model.
    Returns:
        None
    """
    #permet de créer le dossier artifacts s'il n'existe pas déjà, pour stocker les métriques du modèle.
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    metrics = {
        "version": "v1.0",
        "AUC": auc,
        "variables": variables,
        "trained_at": datetime.now().isoformat()
    }
    with open("artifacts/model_meta.json", "w") as f:
        json.dump(metrics, f, indent=2)

def main():

    # Load the dataset
    df = load_data("data/raw.csv")

    # Separate features and target variable
    X, y = separate_target(df)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Preprocess the data (normalization and encoding)
    preprocessor = preprocess_data(X_train)

    # Train the model
    result_pipeline = train_model(X_train, y_train, X_test, y_test, preprocessor)

    # Save the trained model
    save_model(result_pipeline["pipeline"], X_test, y_test)
    save_mertrics(result_pipeline["AUC"], list(X.columns))

main()