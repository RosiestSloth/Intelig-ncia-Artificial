import matplotlib.pyplot as plt
import pandas as pd

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