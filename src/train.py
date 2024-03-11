import os

import numpy as np
# import joblib
import pandas as pd
# import numpy as np
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import TextVectorization, StringLookup, Dense, Dropout
from tensorflow import keras, ragged
from tensorflow.data import AUTOTUNE, Dataset
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# Configuration du cache pour joblib
# cachedir = '/tmp/cache'  # Définissez le répertoire pour le cache
# memory = joblib.Memory(cachedir, verbose=0)


# @memory.cache
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


def build_label_encoder(vocabulary: any) -> StringLookup:
    """
    Crée une couche StringLookup pour encoder les étiquettes.

    Args:
        vocabulary (list): Liste des étiquettes uniques.

    Returns:
        tf.keras.layers.StringLookup: Couche StringLookup adaptée aux étiquettes.

    """
    tf_labels = ragged.constant(np.sort(vocabulary))
    lookup = StringLookup(output_mode="multi_hot")
    lookup.adapt(tf_labels)
    return lookup


def build_text_vectorizer(dataset: pd.DataFrame, max_tokens: int = 1_000_000) -> TextVectorization:
    """
    Crée et adapte une couche TextVectorization.
    """
    vectorize_layer = TextVectorization(
        max_tokens=max_tokens,
        ngrams=None, output_mode="tf_idf",
        standardize=None, split='whitespace'
    )
    vectorize_layer.adapt( dataset.map(lambda text, label: text) )
    return vectorize_layer


def make_dataset(dataset, is_train=True, batch_size=1000, lookup=None):
    """
    Crée un dataset TensorFlow.

    Args:
        dataset (pd.DataFrame): DataFrame contenant les données à utiliser pour le dataset.
        is_train (bool, optional): Indique si le dataset est destiné à l'entraînement. Par défaut, True.
        batch_size (int, optional): Taille des lots pour le dataset. Par défaut, 2000.
        lookup (tf.keras.layers.StringLookup, optional): Couche StringLookup pour encoder les étiquettes.

    Returns:
        tf.data.Dataset: Dataset TensorFlow contenant les données et les étiquettes.
    """

    tf_labels = ragged.constant(dataset['label'].apply(lambda x: [x]).values)
    labels_binarized = lookup(tf_labels) if lookup else tf_labels
    tf_contents = Dataset.from_tensor_slices((dataset['content'].values, labels_binarized))

    if is_train:
        tf_contents = tf_contents.shuffle(batch_size * 100)
    return tf_contents.batch(batch_size)


def make_model(name="language_model", output_shape=22, filepath='../models/best_model_{epoch}.h5'):
    """
    Entraîne un modèle de classification de langues.

    Args:
        name (str, optional): Nom du modèle. Par défaut, "language_model".
        output_shape (int, optional): Nombre de classes de sortie. Par défaut, 22.
        filepath (str, optional): Chemin pour sauvegarder le meilleur modèle. Par défaut, '../models/best_model_{epoch}.keras'.

    Returns:
        tuple: Tuple contenant le modèle, le callback EarlyStopping et le callback ModelCheckpoint.
    """

    model = Sequential(name=name, layers=[
        Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(256, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.5),
        Dense(output_shape, activation="softmax")
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["Accuracy", "Recall", "Precision", "AUC"])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto', restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', save_best_only=True,
                                       verbose=1)

    return model, early_stopping, model_checkpoint


def char_tokenizer(input_text: str) -> str:

    # Supprime tous les chiffres du texte
    cleaned_text = re.sub(r'[\t\n\d ]+', '', input_text.lower())

    # Initialise la chaîne de résultat
    tokenized_text = ''

    # Parcourt chaque caractère du texte nettoyé
    for i, char in enumerate(cleaned_text):
        
        # Ajoute l'espace avant le caractère si nécessaire
        if i>0 and ( tokenized_text[-1] in string.punctuation ) or ( char not in string.punctuation ):
            tokenized_text += ' '
            #print(f"{tokenized_text[-1]}+{char}" )

        # Ajoute le caractère actuel
        tokenized_text += char

    return tokenized_text


if __name__ == "__main__":
    data_dir = '../data/'
    model_dir = '../models/'

    # Utilisation de la fonction mise en cache pour charger et prétraiter les données
    print("Loading and preprocessing data...")
    train_df = load_data(os.path.join(data_dir, 'train_data.tar.bz2'))
    val_df = load_data(os.path.join(data_dir, 'valid_data.tar.bz2'))

    print("Data loaded.")
    print(f"Number of rows in training set: {len(train_df)}")
    print(f"Number of rows in validation set: {len(val_df)}")

    train_df['content'] = train_df['content'].apply(lambda x : char_tokenizer(x))
    val_df['content'] = val_df['content'].apply(lambda x : char_tokenizer(x))

    labels = train_df['label'].unique()
    print(f"Number of labels: {labels.shape}")

    label_encoder = build_label_encoder(labels)

    train_dataset = make_dataset(train_df, is_train=True, lookup=label_encoder)

    valid_dataset = make_dataset(val_df, is_train=False, lookup=label_encoder)

    print("Datasets created.")

    print("Vectorizing...")

    vocabulary_size = 1_000_000

    text_vectorizer = build_text_vectorizer(train_dataset, max_tokens=vocabulary_size)

    print("Vectorization done.")

    train_dataset = train_dataset.map(
        lambda text, label: (text_vectorizer(text), label), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)
    valid_dataset = valid_dataset.map(
        lambda text, label: (text_vectorizer(text), label), num_parallel_calls=AUTOTUNE
    ).prefetch(AUTOTUNE)

    print("Building model...")
    best_model_filepath = model_dir + 'best_model'
    out_shape = label_encoder.vocabulary_size()
    epochs = 20

    model, early_stopping, model_checkpoint = make_model(name="shallow_model"
                                                         , output_shape=out_shape
                                                         , filepath=best_model_filepath
                                                         )

    history = model.fit(train_dataset
                        , validation_data=valid_dataset
                        , epochs=epochs
                        , callbacks=[early_stopping, model_checkpoint]
                        )

    model_final = Sequential(name="langDetect_shallow_model", layers=[
        text_vectorizer,
        model
    ])

    model_final.build()

    print("Saving model...")
    model_final.save(model_dir + 'shallow_model')
    # model_final.save_weights(model_dir + 'shallow_model_weights')

    label_encoder.save_assets(model_dir)

    text_vectorizer.save_assets(model_dir +'/assets/text')
    label_encoder.save_assets(model_dir + '/assets/labels')

    hist_df = pd.DataFrame(history.history)

    hist_df.to_pickle('hist_df.pckl')


    print("loading best model...")
    
    best_model = load_model(best_model_filepath)

    best_model_final = Sequential(name="langDetect_shallow_model", layers=[
        text_vectorizer,
        best_model
    ])
    best_model_final.build()
    print("Saving ...")

    best_model_final.save(model_dir + 'best_shallow_model')

    print("Saving done")



    print("Training done.")
