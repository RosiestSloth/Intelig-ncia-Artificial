import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

"React/ Node.js/ SGBD"

X = np.array([
    [1, 0, 0], # Front End
    [1, 0, 0], # Front End

    [0, 1, 0], # Back End
    [0, 1, 1], # Back End

    [1, 1, 0], # Full Stack
    [1, 1, 1], # Full Stack

    [0, 0, 1] # Data Sicence
])

Y = np.array(['Front End', 'Front End', 'Back End', 'Back End', 'Full Stack', 'Full Stack', 'Data Sience'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

modelo = LinearSVC(random_state=42)
modelo.fit(X_train, Y_train)

dev_novo = np.array([[1, 1, 0]]) # Desenvolvedor Full Stack
previsao = modelo.predict(dev_novo)

print(f"O Desenvolvedor novo é: {previsao}")