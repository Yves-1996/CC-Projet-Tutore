import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf 

st.set_page_config(
    page_title="Détection COVID-19 par CT Scan",
    page_icon="🦠",
    layout="centered"
)

#Charger le Modèle
@st.cache_resource 
def load_model():
    model = tf.keras.models.load_model('my_covid_detection_model.h5')
    return model

model = load_model()

#Prétraitement
def preprocess_image(image):

    # Redimensionner l'image 
    target_size = (64, 64) # Adapte cette taille à l'entrée du modèle
    pil_image = image.convert('RGB')
    pil_image = pil_image.resize(target_size)
    image_array = np.array(pil_image)

    # Normalisation des pixels
    image_array = image_array / 255.0

    # Ajout d'une dimension de batch 
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# Prédiction
def make_prediction(processed_image):
    
    predictions = model.predict(processed_image)

    class_names = ["Indéterminé", "Non-COVID", "COVID" ]
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    result = class_names[predicted_class_index]

    if confidence < 0.7: 
        result = "Indéterminé (Faible confiance < 0.9)"

    return result, confidence

# --- 4. Interface Streamlit ---


st.title("🦠 Détection du COVID-19 à partir d'un CT Scan Pulmonaire")
st.markdown("""
Chargez une image de CT scan pulmonaire et notre système déterminera
si des signes de COVID-19 sont présents, absents, ou si le résultat est indéterminé.
""")

st.warning("⚠️ **Avertissement :** Ce système est à des fins **d'illustration et de démonstration** uniquement. Il ne remplace en aucun cas un diagnostic médical professionnel. Consultez toujours un professionnel de la santé pour un diagnostic précis.")

uploaded_file = st.file_uploader(
    "Veuillez charger une image de CT scan pulmonaire (JPG ou PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Déterminer le type de fichier
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        # Lire l'image standard (JPG, PNG)
        image = Image.open(uploaded_file).convert("L") # Convertir en niveaux de gris
        st.image(image, caption="Image CT Scan", use_container_width=True)
        image_to_process = image

        st.write("Analyse en cours...")
        with st.spinner('Chargement du modèle et prédiction...'):
            processed_image = preprocess_image(image_to_process)
            prediction_result, confidence = make_prediction(processed_image)

        st.success("Analyse terminée !")
        st.write(f"### Résultat : **{prediction_result}**")
        st.write(f"Confiance du modèle : **{confidence:.2f}**")

        if "Non-COVID" in prediction_result:
            st.info("Aucun signe de COVID-19 n'a été détecté (basé sur l'analyse de l'image")
        elif "COVID" in prediction_result:
            st.error("Des signes potentiels de COVID-19 ont été détectés.")
        else:
            st.warning("Le résultat est indéterminé. Cela peut être dû à une faible confiance du modèle ou à des caractéristiques ambiguës.")

    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement de l'image : {e}")
        st.info("Veuillez vous assurer que le fichier est un CT scan valide et que votre fonction `preprocess_image` est correctement configurée.")

st.markdown("---")
