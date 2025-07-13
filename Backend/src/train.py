import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import limpiar_texto  

# 1. se carga del dataset limpio
df = pd.read_csv('../data/reviews_limpio.csv', encoding='utf-8')

# 2. variables para entrenar al modelo
X = df['clean_text']
y = df['label']

# 3. se verifica que no hayan nulos
assert not df.isnull().values.any(), "El dataset contiene valores nulos"

# 4. división de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5.vectorización TF-IDF con n-gramas
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6.aplicación del modelo Logistic Regression
model = LogisticRegression(max_iter=300, random_state=42)
model.fit(X_train_vec, y_train)

# 7. Evaluación
preds = model.predict(X_test_vec)
print("\n📊 Accuracy:", accuracy_score(y_test, preds))
print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, preds))
print("📈 Classification Report:\n", classification_report(y_test, preds))

# 8. guardamos el modelo
with open('../models/sentiment_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("\n✅ Modelo guardado en ../models/sentiment_model.pkl")

# 9. prueba con frases realistas
test_phrases = [
    "El producto es excelente y me encantó",
    "Muy decepcionado, no funciona como esperaba",
    "La calidad es pésima, no lo recomiendo",
    "Servicio al cliente fantástico y rápido",
    "No me gustó, llegó dañado y mal embalado",
    "Estoy muy satisfecho con la compra, la recomiendo",
    "Terrible experiencia, no volveré a comprar",
    "Muy buena calidad, vale la pena",
    "El peor producto que he comprado",
    "Cumple con lo que promete, muy bien"
]

X_test_manual = vectorizer.transform([limpiar_texto(p) for p in test_phrases])
preds_manual = model.predict(X_test_manual)

print("\n📝 Clasificación de ejemplos realistas:")
for phrase, pred in zip(test_phrases, preds_manual):
    print(f"  '{phrase}': {'Positivo 😊' if pred == 1 else 'Negativo 😞'}")
