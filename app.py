from flask import Flask, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Carrega o modelo e o scaler
modelo = load_model("modelo_heart_failure.h5")
scaler = joblib.load("scaler.pkl")
colunas_modelo = joblib.load("colunas_modelo.pkl")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Entrada: JSON com os dados brutos
        dados = request.json

        print("=== DADOS BRUTOS ===")
        print(dados)

        # 1. Corrigir valores
        oldpeak = float(dados.get("Oldpeak", 0))
        oldpeak = max(oldpeak, 0)  # zera valores negativos

        cholesterol = float(dados.get("Cholesterol", 0))
        if cholesterol == 0:
            cholesterol = 223

        # 2. Mapear variáveis categóricas simples
        sex = 1 if dados.get("Sex") == "M" else 0
        exercise_angina = 1 if dados.get("ExerciseAngina") == "Y" else 0

        fasting_bs = int(dados.get("FastingBS", 0))

        # 3. One-hot encoding manual (com drop_first=True)
        # Se categoria for a "dropada", todas dummies = 0

        # ChestPainType: dropada = ASY
        chestpain_map = {
            "ATA": [1, 0, 0],
            "NAP": [0, 1, 0],
            "TA":  [0, 0, 1],
            "ASY": [0, 0, 0]
        }
        chestpain_encoded = chestpain_map.get(dados.get("ChestPainType"), [0, 0, 0])

        # RestingECG: dropada = Normal
        restingecg_map = {
            "Normal": [1, 0],
            "ST":     [0, 1],
            "LVH":    [0, 0]
        }
        restingecg_encoded = restingecg_map.get(dados.get("RestingECG"), [0, 0])


        # ST_Slope: dropada = Down
        st_slope_map = {
            "Flat": [1, 0],
            "Up":   [0, 1],
            "Down": [0, 0]
        }
        st_slope_encoded = st_slope_map.get(dados.get("ST_Slope"), [0, 0])

        # 4. Coletar os dados numéricos
        numericos = [
            float(dados.get("Age", 0)),
            float(dados.get("RestingBP", 0)),
            cholesterol,
            float(dados.get("MaxHR", 0)),
            oldpeak
        ]

        # 5. Aplicar o scaler (somente nos 5 primeiros)
        numericos_array = np.array(numericos).reshape(1, -1)
        numericos_normalizados = scaler.transform(numericos_array).flatten().tolist()

        # entrada_dict é um dicionário com chave -> valor processado
        entrada_dict = {
            "Age": numericos_normalizados[0],
            "Sex": sex,
            "RestingBP": numericos_normalizados[1],
            "Cholesterol": numericos_normalizados[2],
            "FastingBS": fasting_bs,
            "MaxHR": numericos_normalizados[3],
            "ExerciseAngina": exercise_angina,
            "Oldpeak": numericos_normalizados[4],
            "ChestPainType_ATA": chestpain_encoded[0],
            "ChestPainType_NAP": chestpain_encoded[1],
            "ChestPainType_TA": chestpain_encoded[2],
            "RestingECG_Normal": restingecg_encoded[0],
            "RestingECG_ST": restingecg_encoded[1],
            "ST_Slope_Flat": st_slope_encoded[0],
            "ST_Slope_Up": st_slope_encoded[1]
        }

        # Ordena conforme o que foi salvo no Colab
        entrada_modelo = [entrada_dict[col] for col in colunas_modelo]

        # Converte pra array e faz a previsão
        entrada_array = np.array(entrada_modelo).reshape(1, -1)


        # 🖨️ Imprimir todos os valores processados
        print("=== ENTRADA PROCESSADA ===")
        print("Entrada Modelo:", entrada_modelo)
        print("==========================")

        # 7. Previsão
        pred = modelo.predict(entrada_array)[0][0]
        resultado = "Risco de falha cardíaca" if pred > 0.5 else "Sem risco"

        return jsonify({
            "resultado": resultado,
            "probabilidade": round(float(pred), 4)
        })

    except Exception as e:
        return jsonify({"erro": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
