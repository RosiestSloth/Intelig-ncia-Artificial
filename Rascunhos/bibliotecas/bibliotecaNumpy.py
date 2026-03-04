#Biblioteca Numpy
import numpy as  np

array_1d = np.arange(1, 11)

array_2d = np.random.rand(3, 3)

media = np.mean(array_1d)

print(f'A média é: {media}')

media = np.mean(array_2d)

print(f'A média é: {media}')

soma = np.sum(array_1d)

print(f'A soma é: {soma}')

