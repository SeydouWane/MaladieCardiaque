from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle SVM
with open("meilleur_svm.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def index():
    return render_template("index.html", title="Accueil")

# Fonction pour générer des conseils basés sur la prédiction
def generate_advice(prediction):
    if prediction == 0:
        return [
            {"label": "Adoptez une alimentation équilibrée", "url": "https://cndn.sn/comprendre-la-nutrition/la-nutrition-au-senegal/defis-de-la-nutrition-au-senegal/"},
            {"label": "Continuez à pratiquer des exercices réguliers", "url": "https://clubsport-bienetre.com/"}
        ]
    else:
        return [
            {"label": "Consultez un cardiologue immédiatement", "url": "https://www.med.tn/docteur-senegal/cardiologue/dakar"},
            {"label": "Réduisez votre consommation de sel", "url": "https://example.com/sel"},
            {"label": "Envisagez un programme de réadaptation cardiaque", "url": "https://clubsport-bienetre.com/"}
        ]

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            data = request.form
            input_data = np.array([[
                int(data['age']),
                int(data['sex']),
                int(data['cp']),
                int(data['trestbps']),
                int(data['chol']),
                int(data['fbs']),
                int(data['restecg']),
                int(data['thalach']),
                int(data['exang']),
                float(data['oldpeak']),
                int(data['slope']),
                float(data['ca']),
                int(data['thal'])
            ]])
            prediction = model.predict(input_data)[0]
            result = "Pas de maladie cardiaque détectée." if prediction == 0 else f"Maladie détectée cardiaque (Niveau {prediction})."
            advice = generate_advice(prediction)
            return jsonify({"status": "success", "result": result, "advice": advice})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    return render_template("predict.html", title="Prédiction")


@app.route("/calculate_img", methods=["GET", "POST"])
def calculate_img():
    if request.method == "POST":
        try:
            data = request.form
            poids = float(data['poids'])
            taille = float(data['taille'])
            age = int(data['age'])
            sexe = int(data['sexe'])
            imc = poids / (taille ** 2)
            img = (1.20 * imc) + (0.23 * age) - (16.2 if sexe == 1 else 5.4)
            if img < 18:
                orientation = "Insuffisance de masse grasse."
            elif 18 <= img < 25:
                orientation = "Masse grasse normale."
            elif 25 <= img < 30:
                orientation = "Surpoids."
            else:
                orientation = "Obésité. Consultez un professionnel."
            return jsonify({"status": "success", "img": img, "orientation": orientation})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})
    return render_template("img.html", title="Calcul IMG")

if __name__ == "__main__":
    app.run(debug=True)
