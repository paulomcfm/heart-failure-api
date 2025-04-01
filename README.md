# 🫀 API - Predição de Risco de Doença Cardíaca

Esta é uma API em Flask que utiliza um modelo de Machine Learning (rede neural) treinado para prever o risco de doença cardíaca com base em dados clínicos do paciente. O modelo foi desenvolvido com TensorFlow/Keras e utiliza entrada formatada em JSON.

## 📦 Tecnologias utilizadas

- Python 3.11+
- Flask
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy
- Flask-CORS

## 🔥 Como rodar o projeto localmente

### 1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/api-predicao-cardio.git
cd api-predicao-cardio
```

### 2. Instale as dependências:
Recomenda-se usar um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
pip install -r requirements.txt
```

### 3. Rode a API:
```bash
python app.py
```

A API irá iniciar em `http://127.0.0.1:5000`

## 🧪 Endpoint principal

### `POST /predict`
Recebe dados do paciente no formato JSON:
```json
{
  "Age": 48,
  "Sex": "F",
  "ChestPainType": "ASY",
  "RestingBP": 138,
  "Cholesterol": 214,
  "FastingBS": 0,
  "RestingECG": "Normal",
  "MaxHR": 108,
  "ExerciseAngina": "Y",
  "Oldpeak": 1.5,
  "ST_Slope": "Flat"
}
```

E retorna:
```json
{
  "resultado": "Risco de falha cardíaca",
  "probabilidade": 0.8531
}
```

## 🧠 Lógica de predição
- O modelo recebe os dados brutos, trata e normaliza internamente.
- Realiza `one-hot encoding` manual de variáveis categóricas.
- Utiliza `StandardScaler` previamente salvo para normalização.
- Aplica o modelo `.h5` com Keras para obter a probabilidade da classe positiva.
- Um limiar de 0.5 é usado para decidir entre "Risco" ou "Normal".

## 📁 Arquivos importantes
- `modelo_heart_failure.h5` - modelo treinado
- `scaler.pkl` - normalizador usado durante o treinamento
- `colunas_modelo.pkl` - ordem dos atributos esperados pelo modelo

## 🌐 Front-end integrado
Um front-end React está sendo desenvolvido para consumo da API. Veja o repositório correspondente: [https://github.com/paulomcfm/heart-failure]

## 📄 Licença
Este projeto é de uso educacional/livre. Fique à vontade para adaptar.

