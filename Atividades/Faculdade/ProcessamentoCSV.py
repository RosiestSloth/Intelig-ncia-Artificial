import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

def avaliar_modelo(modelo, x_data, y_data, nome="Modelo"):
  x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42, stratify=y_data)
    
  modelo.fit(x_train, y_train)
  y_pred = modelo.predict(x_test)

  print(f'\n===== {nome} =====')
  print(f'A Acurácia é: {accuracy_score(y_test, y_pred)*100:.2f}%')
  print("Relatório de Classificação:")
  print(classification_report(y_test, y_pred))
  print("Matriz de Confusão:")
  print(confusion_matrix(y_test, y_pred))


# importar ele no aprendizado de máquina
# Verificar a acurácia

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

# plt.scatter(df['sepal_length'], df['sepal_width'])
# plt.xlabel('comprimento da Sépala')
# plt.ylabel('largura da Sépala')
# plt.title('Gráfico de Dispersão: Comprimento X Largura')
# plt.show()

plt.hist(df['petal_length'])
plt.show()

# 1. Separando as features (X) e o target (y)
X_iris = df.drop('species', axis=1)
y_iris = df['species']

# 2. Divisão de treino e teste
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42
)

# 3. Treinando o modelo
modelo_iris = LinearSVC(random_state=42, max_iter=10000) # Aumentar iter para convergência
modelo_iris.fit(X_train_iris, y_train_iris)

# 4. Fazendo previsões e calculando a acurácia
previsoes_iris = modelo_iris.predict(X_test_iris)
acuracia = accuracy_score(y_test_iris, previsoes_iris)

avaliar_modelo(modelo_iris, X_iris, y_iris)

