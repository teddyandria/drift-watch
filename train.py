#etape 1 : charger le csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


df = pd.read_csv("data/raw.csv")

#séparer la variable cible du reste: la cible est la colonne "class"
X = df.drop(columns=["class"])
y = df["class"]

#conversion binaire. (pas de boucle lorsqu'on manipule pandas, on peut utiliser la méthode map pour transformer les valeurs de la colonne "class" en 0 et 1)
y = y.map({"bad": 1, "good": 0})

#taux de défaut moyen
# print(y.mean()) 


#étape 2 : split du dataset en train et test (80/20). stratifiy permet de conserver la même proportion de classes dans les deux datasets. random_state permet de reproduire le split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

#Construis un préprocesseur qui applique un encodage one-hot aux colonnes texte et une standardisation aux colonnes numériques, en une seule opération.

#etape 3 : Prétraitement des données
#ColumnTransformer permet de transformer et traiter les variables afin d'avoir un dataset prêt pour l'entrainement du modèle.
numerical_features = X.select_dtypes(include=["number"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# print(f"Numerical features: {len(numerical_features)}")
# print(f"Categorical features: {len(categorical_features)}")

#standardScaler permet de mettre à la même échelle les variables numériques/
#oneHotEncoder permet de transformer les variables catégorielles en variables binaires (0 ou 1) pour chaque modalité.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features) 
    ]
)

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