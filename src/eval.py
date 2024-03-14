import os

from tensorflow.keras.layers import StringLookup
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow import strings,keras
import pandas as pd


# from train import load_data
@keras.saving.register_keras_serializable()
def standardize_text(input_text: str) -> str:
    """
    Standardise un texte donné.

    Args:
        input_text (str): Texte à standardiser.

    Returns:
        str: Texte standardisé.
    """
    return strings.regex_replace(input_text, '[\n\t ]+', '', replace_global=True)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Charge et prétraite les données.

    Args:
        data_path (str): Chemin vers le fichier de données.

    Returns:
        pd.DataFrame: DataFrame contenant les données chargées et prétraitées.
    """
    df = pd.read_pickle(data_path)
    df.reset_index(drop=True, inplace=True)
    return df


def build_label_encoder(assets_dir: str) -> StringLookup:
    """
    Crée une couche StringLookup pour encoder les étiquettes.

    Args:
        assets_dir (str): Chemin vers le répertoire où sauvegarder les assets.

    Returns:
        tf.keras.layers.StringLookup: Couche StringLookup adaptée aux étiquettes.

    """

    lookup = StringLookup(output_mode="multi_hot")
    lookup.load_assets(assets_dir)
    return lookup


if __name__ == "__main__":
    data_dir = '../data/'
    model_dir = '../models/'

    test_data = load_data(os.path.join(data_dir, 'test_data.tar.bz2'))

    print(f"Number of rows in test set: {len(test_data)}")

    print("Loading model...")

    model = models.load_model(model_dir + 'best_shallow_model.keras')
    print("Model loaded.")
    predicted_probabilities = model.predict(['un test du modèle par la prédiction de la probabilité de chaque classe.'])

    print("Done.")
    # afficher version tensorflow

    new_lookup = StringLookup(output_mode="multi_hot")
    new_lookup.load_assets('../models/labels_assets')

    r2 = [
        x
        for _, x in sorted(
            zip(predicted_probabilities[0], new_lookup.get_vocabulary()),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ]
    print(r2[:5])
