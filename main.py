import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Carga el dataset
datos = pd.read_csv("zoo3.csv")

# Maneja valores faltantes (si los hay)
# ... (Implementación de manejo de valores faltantes)

# Codificación de etiquetas
le = LabelEncoder()
datos["animal_name"] = le.fit_transform(datos["animal_name"])
datos["class_type"] = le.fit_transform(datos["class_type"])

# Selección de características (opcional)
# ... (Implementación de selección de características)

# Divide los datos en conjuntos de entrenamiento y prueba
X = datos.drop("class_type", axis=1)  # Características
y = datos["class_type"]  # Variable objetivo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementa y evalúa los modelos
def evaluar_modelo(nombre_modelo, modelo):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    precisión = accuracy_score(y_test, y_pred)
    exactitud = precision_score(y_test, y_pred, average="weighted")
    sensibilidad = recall_score(y_test, y_pred, average="weighted")
    especificidad = 1 - sensibilidad  # Suponiendo clasificación binaria
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"Resultados del {nombre_modelo}:")
    print(f"\tPrecisión (Accuracy): {precisión:.4f}")
    print(f"\tExactitud (Precision): {exactitud:.4f}")
    print(f"\tSensibilidad (Recall): {sensibilidad:.4f}")
    print(f"\tEspecificidad: {especificidad:.4f}")
    print(f"\tPuntaje F1: {f1:.4f}")
    print("-" * 50)

# Regresión Logística
modelo_lr = LogisticRegression()
evaluar_modelo("Regresión Logística", modelo_lr)

# K-Vecinos Cercanos (KNN)
modelo_knn = KNeighborsClassifier(n_neighbors=5)  # Ajusta hiperparámetros según sea necesario
evaluar_modelo("K-Vecinos Cercanos (KNN)", modelo_knn)

# Máquinas Vector Soporte (SVM)
modelo_svm = SVC(kernel="linear")  # Ajusta hiperparámetros según sea necesario
evaluar_modelo("Máquinas Vector Soporte (SVM)", modelo_svm)

# Naive Bayes
modelo_nb = GaussianNB()
evaluar_modelo("Naive Bayes", modelo_nb)