import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

"Características das frutas [Arredondada, Superfície rugosa, ácida]"
X = np.array([
    [1, 0, 0], # Maçã
    [1, 0, 0], # Maçã

    [1, 1, 1], # Laranja
    [0, 1, 1] # Laranja
])

Y = np.array(['maçã', 'maçã', 'laranja', 'laranja'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

modelo = LinearSVC(random_state=42)
modelo.fit(X_train, Y_train)

fruta_nova = np.array([[1, 1, 0]])
previsao = modelo.predict(fruta_nova)

print(f'A fruta provavelmente é uma {previsao}')
