# ALPHA_KEITA_M2_DEV_GROUPE1

## Description

Ce projet est une application de détection de mélanome utilisant l'apprentissage profond et une API pour entraîner un modèle et prédire des images. Il inclut également un chatbot utilisant GPT-3.5 d'OpenAI.

## Fonctionnalités

- **Entraînement de modèle** : Permet d'entraîner un modèle de détection de mélanome à partir d'images fournies.
- **Prédiction** : Utilise le modèle entraîné pour prédire si une image donnée est susceptible de représenter un mélanome.
- **Chatbot** : Un chatbot intégré pour fournir des informations supplémentaires ou répondre à des questions.

## Technologies utilisées

- FastAPI pour la création de l'API.
- TensorFlow et Keras pour l'entraînement des modèles de machine learning.
- Streamlit pour l'interface utilisateur.
- OpenAI pour le chatbot.

## Installation

Pour installer et exécuter ce projet, suivez ces étapes :

1. Clonez le dépôt :

```
git clone https://github.com/alphakta/sample_api_IA_ML.git
```

2. Installez les dépendances :

```
pip install -r requirements.txt
```

3. Lancez l'API :

```
uvicorn api:app --reload
```

4. Ouvrez un autre terminal et lancez l'application Streamlit :

```
streamlit run app.py
```

## Utilisation

- Pour entraîner le modèle, utilisez le bouton "Train Model" dans l'application Streamlit & créer un dossier training dans le répertoire avec dedands les différentes images qui devra être utiliser pour entrainement...
- Pour faire une prédiction, téléchargez une image via l'interface Streamlit et le résultat s'affichera.
