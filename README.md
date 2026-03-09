# Inteligência Artificial - Estudos e Atividades

Repositório com exercícios, rascunhos e práticas de **Inteligência Artificial** e **Ciência de Dados** usando Python.

O foco deste projeto é treinar fundamentos como:

- classificação supervisionada com `scikit-learn`;
- manipulação de dados com `pandas`;
- operações numéricas com `numpy`;
- visualização com `matplotlib`.

## Estrutura do projeto

```text
Atividades/
  Estudo/
    Desenvolvedores.py
    Frutos.py
  Faculdade/
    ProcessamentoCSV.py
    ReadCSV.py
Rascunhos/
  bibliotecas/
    bibliotecaMatplotlib.py
    bibliotecaNumpy.py
    bibliotecaPandas.py
  IA/
    TreinamentoIA.py
```

## O que cada script faz

### Atividades/Estudo

- **Desenvolvedores.py**  
  Treina um classificador `LinearSVC` para prever perfil de desenvolvedor (Front End, Back End, Full Stack, Data Science) com base em características binárias.

- **Frutos.py**  
  Treina um `LinearSVC` para classificar fruta (maçã/laranja) a partir de características simples.

### Atividades/Faculdade

- **ReadCSV.py**  
  Lê o dataset Iris via URL com `pandas`, exibindo amostra, informações e estatísticas.

- **ProcessamentoCSV.py**  
  Faz leitura do Iris, visualização de histograma, treino de modelo `LinearSVC` e avaliação com acurácia, relatório de classificação e matriz de confusão.

### Rascunhos/bibliotecas

- **bibliotecaNumpy.py**  
  Exemplos básicos com arrays, média e soma em `numpy`.

- **bibliotecaPandas.py**  
  Exemplos iniciais de leitura e inspeção de dados com `pandas`.

- **bibliotecaMatplotlib.py**  
  Exemplo de visualização (histograma) com `matplotlib`.

### Rascunhos/IA

- **TreinamentoIA.py**  
  Estudo mais completo de classificação binária (porco x cachorro), incluindo:
  - avaliação de modelo (`accuracy`, relatório e matriz de confusão);
  - baseline com `DummyClassifier`;
  - teste com dados aleatórios;
  - expansão sintética de dataset para comparação de desempenho.

## Tecnologias e bibliotecas

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Como executar

### 1) Clonar o repositório

```bash
git clone https://github.com/RosiestSloth/Intelig-ncia-Artificial
cd Intelig-ncia-Artificial
```

### 2) Criar e ativar ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Instalar dependências

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 4) Rodar scripts

Exemplos:

```bash
python Atividades/Estudo/Frutos.py
python Atividades/Faculdade/ReadCSV.py
python Atividades/Faculdade/ProcessamentoCSV.py
python Rascunhos/IA/TreinamentoIA.py
```

## Observações

- Alguns scripts usam dataset remoto (Iris) via URL; é necessário acesso à internet para execução.
- O repositório contém arquivos de estudo e experimentação, então partes do código podem estar em evolução.

## Próximos passos sugeridos

- Criar um `requirements.txt` para facilitar instalação.
- Padronizar nomes e corrigir pequenos typos de classes/labels.
- Adicionar mais exemplos de métricas e validação cruzada.

---

Feito para fins de estudo e prática acadêmica.