import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier

# ==========================================
# 4. FUNÇÃO DE AVALIAÇÃO DE DESEMPENHO
# ==========================================

def avaliar_modelo(modelo, x_data, y_data, nome="Modelo"):
    """
    Realiza o split, treinamento e exibe métricas de performance.
    - Acurácia: Porcentagem de acertos totais.
    - Relatório de Classificação: Precision, Recall e F1-Score.
    - Matriz de Confusão: Mostra onde o modelo confundiu as classes.
    """
    # stratify=y_data mantém a proporção das classes no treino e teste
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42, stratify=y_data)
    
    modelo.fit(x_train, y_train)
    y_pred = modelo.predict(x_test)

    print(f'\n===== {nome} =====')
    print(f'A Acurácia é: {accuracy_score(y_test, y_pred)*100:.2f}%')
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))


# ==========================================
# 6. EXPANSÃO DO DATASET (SINTÉTICO ESTRUTURADO)
# ==========================================
0
def expandir_cachorros(n):
    """Gera dados sintéticos que seguem o padrão de um cachorro."""
    dados = []
    for _ in range(n):
        dados.append([
            1, # Tem pelo
            1, # Late
            np.random.randint(0, 2), # Perna longa (variável)
            1, # Focinho achatado
            np.random.randint(0, 2)  # Rabo enrolado (variável)
        ])
    return np.array(dados)

def expandir_porcos(n):
    """Gera dados sintéticos que seguem o padrão de um porco."""
    dados = []
    for _ in range(n):
        dados.append([
            1, # Tem pelo
            0, # Não late
            0, # Não tem perna longa
            1, # Focinho achatado
            np.random.randint(0, 2) # Rabo enrolado (variável)
        ])
    return np.array(dados)


# ==========================================
# 1. DEFINIÇÃO DOS DADOS INICIAIS (DATASET)
# ==========================================

# Características (Features): [tem_pelo, late, perna_longa, focinho_achatado, rabo_enrolado]
# Representação binária: 1 = Sim, 0 = Não
X = np.array([
    [1, 0, 0, 1, 1], # Porco
    [0, 0, 0, 1, 1], # Porco
    [1, 0, 0, 1, 0], # Porco

    [1, 1, 1, 0, 0], # Cachorro
    [1, 1, 1, 0, 1], # Cachorro
    [1, 1, 0, 1, 0], # Cachorro
])

# Rótulos (Labels): As respostas corretas para o treinamento
Y = np.array(['porco', 'porco', 'porco', 'cachorro', 'cachorro', 'cachorro'])

# ==========================================
# 2. TREINAMENTO DO MODELO LINEAR SVC
# ==========================================

# Divisão de treino e teste para validar a capacidade de generalização
# test_size=0.3 separa 30% dos dados para teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# LinearSVC: Classificador que busca um hiperplano para separar as classes
modelo = LinearSVC(random_state=42)
modelo.fit(X_train, Y_train) # Ajusta o modelo aos dados de treino

# ==========================================
# 3. TESTES DE PREDIÇÃO INDIVIDUAL
# ==========================================

# Teste 1: Animal com características de Porco
animal_novo = np.array([[1, 0, 0, 1, 1]])
previsao = modelo.predict(animal_novo)
print(f'Possivelmente o animal é um: {previsao}')

# Teste 2: Animal com características mistas
animal_novo_2 = np.array([[1, 1, 1, 1, 0]])
previsao_2 = modelo.predict(animal_novo_2)
print(f'Possivelmente o animal é um: {previsao_2}')

# Avaliação inicial com os dados pequenos
avaliar_modelo(modelo, X, Y, nome="LinearSVC (Dados Originais)")

# DummyClassifier: Um modelo "burro" que serve de baseline (base de comparação)
dummy = DummyClassifier(strategy="stratified", random_state=42)
avaliar_modelo(dummy, X, Y, nome="Dummy Baseline (Estratégico)")

# ==========================================
# 5. TESTE COM DADOS ALEATÓRIOS (RUÍDO)
# ==========================================

np.random.seed(42)

# Criando 100 amostras com características e rótulos totalmente aleatórios
X_random = np.random.randint(0, 2, (100, 5))
Y_random = np.random.choice(['porco', 'cachorro'], size=100)

print("\n--- TESTE COM DADOS ALEATÓRIOS ---")
avaliar_modelo(modelo, X_random, Y_random, nome="LinearSVC (Aleatório)")
avaliar_modelo(dummy, X_random, Y_random, nome="Dummy (Aleatório)")

# Parâmetros de expansão
n_porco = 100
n_cachorro = 60

porcos_exp = expandir_porcos(n_porco)
cachorros_exp = expandir_cachorros(n_cachorro)

# Unindo os dados originais com os dados expandidos
X_exp_struct = np.vstack((X, porcos_exp, cachorros_exp))
Y_exp_struct = np.concatenate((Y, ['porco']*n_porco, ['cachorro']*n_cachorro))

# Avaliação final com dataset maior e mais consistente
print("\n--- TESTE COM DATASET EXPANDIDO ---")
avaliar_modelo(modelo, X_exp_struct, Y_exp_struct, nome="LinearSVC (Expandido)")
avaliar_modelo(dummy, X_exp_struct, Y_exp_struct, nome="Dummy (Expandido)")