# langDetect shallow model
## À propos du projet

Le projet `langDetect shallow model` est une implémentation de détection de langue qui utilise des techniques d'apprentissage profond pour identifier la langue d'un texte donné. Basé sur TensorFlow et Keras, ce modèle est capable de reconnaître plusieurs langues avec une haute précision. Le projet comprend des scripts pour le prétraitement des données (`preprocess_data.py`) et l'entraînement du modèle (`train.py`).

## Configuration requise

- Python 3.10 ou plus récent
- TensorFlow 2.15.0
- Pandas 2.0.0 ou plus récent
- Scikit-learn
- Numpy

## Installation

### Exécution Locale avec un Environnement Virtuel
Il est recommandé d'utiliser un environnement virtuel Python.




1. **Cloner le dépôt** :

   ```
   git clone https://github.com/LangDetectAPI/model_langDetect.git
   ```
2. **Créer un environnement virtuel** :

   Naviguez jusqu'au répertoire racine de votre projet, puis exécutez la commande suivante pour créer un environnement virtuel :

   ```bash
   python3 -m venv .venv
   ```

3. **Activer l'environnement virtuel** :

   Pour activer l'environnement virtuel, exécutez :
   ```bash
     .venv\Scripts\activate
   ```

4. **Installer les dépendances** :

   Dans le répertoire du projet, exécutez :

   ```
   pip install -r requirements.txt
   ```

## Téléchargement des données

Avant de pouvoir exécuter le script de prétraitement, vous devez télécharger le fichier des phrases depuis le site de Tatoeba :

1. Visitez la page [Téléchargements de Tatoeba](https://tatoeba.org/fr/downloads).
2. Téléchargez le fichier `sentences.tar.bz2` et placez-le dans le répertoire `data/` de votre projet.

## Préparation des données

Utilisez le script `preprocess_data.py` pour nettoyer et préparer les données pour l'entraînement. Assurez-vous que vos données sont placées correctement comme décrit ci-dessus.

Pour exécuter le script de prétraitement :

```
python preprocess_data.py --data_dir ./data --corpus_path ./data/sentences.tar.bz2 --macrolanguages_path ./data/iso-639-3-macrolanguages.tab
```

## Entraînement du modèle

Après avoir prétraité les données, utilisez le script `train.py` pour entraîner le modèle. Ce script chargera les données d'entraînement et de validation, construira le modèle, et l'entraînera.

Pour démarrer l'entraînement :

```
python train.py
```

## Utilisation du modèle

Une fois l'entraînement terminé, le modèle sera sauvegardé dans le répertoire `models/`. Vous pouvez charger ce modèle pour prédire la langue de nouveaux textes.

## Contribution

Les contributions au projet sont les bienvenues. Si vous souhaitez contribuer, veuillez suivre les étapes suivantes :

1. Forker le dépôt.
2. Créer une nouvelle branche pour votre fonctionnalité.
3. Soumettre une pull request.

## Licence

Ce projet est distribué sous licence MIT. Pour plus d'informations, voir le fichier `LICENSE`.

---
