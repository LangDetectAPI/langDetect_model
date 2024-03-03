import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

__all__ = ['preprocess_data']


def preprocess_data(data_dir, corpus_path, macrolanguages_path, labels, test_split=0.2, nrows=10_000):
    """Prétraitement des données

    Args:
        data_dir (str): Répertoire de sauvegarde des données
        corpus_path (str): Chemin du fichier contenant les données
        macrolanguages_path (str): Chemin du fichier contenant les langues
        labels (list): Liste des langues à garder
        test_split (float): Proportion des données à garder pour le test
        nrows (int): Nombre de lignes à garder pour chaque label

    Returns:
        train_data (pd.DataFrame): Données d'entraînement
        test_data (pd.DataFrame): Données de test



        """

    # Vérifier si les fichiers existent
    if not os.path.exists(corpus_path) or not os.path.exists(macrolanguages_path):
        print("Un ou plusieurs fichiers sont manquants.")
        return

    # Chargement du corpus
    print(f"Load corpus from file {corpus_path}...")

    data = pd.read_csv(corpus_path, sep='\t', header=None, compression='bz2', low_memory=False)
    data.columns = ['id', 'I_Id', 'content']

    # suppression des valeurs NAN
    data.dropna(subset=["I_Id"], inplace=True)

    # chargement du fichier macrolanguages
    print(f"Load macrolanguages file {macrolanguages_path}...")
    macro_lang = pd.read_csv(macrolanguages_path, sep='\t')
    macro_lang.columns = ['label', 'I_Id', 'I_Status']

    print(f"Preprocessing start...")

    # jointure entre la table des sentences et la table macrolanguages
    data = pd.merge(data, macro_lang, how='left', on='I_Id')

    # Remplacer les valeurs NaN dans la colonne 'lang' par la valeur de la colonne 'I_Id'
    data['label'] = data['label'].fillna(data['I_Id'])

    # suppression des columns 'I_Status', 'id' et 'I_Id'
    data = data.drop(columns=['I_Status', 'id', 'I_Id'])

    # garder que les langues selectionnées dans la liste labels
    data = data[data['label'].isin(set(labels))]

    # suppression des doublons
    data = data[~data["content"].duplicated()]

    #
    # Rééchantillonnage des données
    #
    #
    # Étape 1: Identifier les labels avec plus de 100000 lignes
    #
    print("Rééchantillonnage des données...")

    # Compter le nombre de lignes par label après le filtrage initial
    label_counts = data['label'].value_counts()

    # Identifier les labels avec plus de n lignes

    labels_over_nrows = label_counts[label_counts > nrows].index.tolist()

    print(f"labels over {nrows} rows: {labels_over_nrows}")

    #
    # Étape 2: Filtrer les lignes avec moins de 4 tokens uniquement pour ces labels
    #

    # Liste des labels à exclure du filtrage
    labels_to_exclude = ['jpn', 'zho', 'hin', 'kor', 'vie', 'tha']

    # Filtrer la liste des labels_over_nrows pour exclure certains labels
    labels_for_filtering = [label for label in labels_over_nrows if label not in labels_to_exclude]

    # Filtrer df pour ne garder que les lignes des labels sélectionnés pour le filtrage
    df_for_filtering = data[data['label'].isin(labels_for_filtering)]

    print("filtering...")

    # Appliquer le filtrage basé sur le nombre de tokens uniquement sur ces lignes
    min_token = 4

    df_filtered = df_for_filtering[df_for_filtering['content'].apply(lambda x: len(x.split()) >= min_token)]

    # Ajouter les lignes des labels à exclure du filtrage sans les filtrer
    df_excluded_labels = data[data['label'].isin(labels_to_exclude)]

    # Ajouter les lignes des labels qui n'ont pas plus de 100000 lignes et donc n'ont pas été filtrées
    df_under_100k = data[~data['label'].isin(labels_over_nrows)]

    # Concaténer les trois ensembles pour obtenir le dataframe final
    df_final = pd.concat([df_filtered, df_excluded_labels, df_under_100k], axis=0)

    #
    # Étape 4: Limiter à 100000 lignes pour les labels filtrés
    #

    print(f"Limiting to {nrows} rows for each label...")

    # Initialiser un DataFrame vide pour le résultat final limité
    data_final = pd.DataFrame()

    # Limiter à 100000 lignes pour chaque label concerné
    for label in labels_over_nrows:
        df_temp = df_final[df_final['label'] == label].sample(n=nrows, random_state=42)
        data_final = pd.concat([data_final, df_temp], axis=0)

    # Ajouter les lignes des labels non concernés par la limitation
    data_final = pd.concat([data_final, df_final[~df_final['label'].isin(labels_over_nrows)]], axis=0)

    print(f"Final data shape: {data_final.shape}")

    # Sauvegarder le résultat dans un fichier

    test_split = 0.2

    # Initial train and test split.
    train_data, test_data = train_test_split(
        data_final,
        test_size=test_split,
        stratify=data_final['label'].values,
    )

    val_split = 0.5
    valid_data = test_data.sample(frac=val_split)
    test_data.drop(valid_data.index, inplace=True)

    # reindexing
    train_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)

    print(f"Preprocessing end...")

    # Sauvegarder les résultats dans des fichiers
    print(f"Sauvegarde des résultats dans '{data_dir}*_data.pkl'...")
    train_data.to_pickle(path=os.path.join(data_dir, 'train_data.tar.bz2'))
    test_data.to_pickle(os.path.join(data_dir, 'test_data.tar.bz2'))
    valid_data.to_pickle(os.path.join(data_dir, 'valid_data.tar.bz2'))

    return train_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data")
    parser.add_argument('--data_dir', type=str
                        , default='../data/'
                        , help="Répertoire de sauvegarde des données")
    parser.add_argument('--corpus_path'
                        , type=str
                        , default='../data/sentences.tar.bz2',
                        help="Chemin du fichier contenant les données")
    parser.add_argument('--macrolanguages_path'
                        , type=str
                        , default='../data/iso-639-3-macrolanguages.tab'
                        , help="Chemin du fichier contenant les langues")
    parser.add_argument('--test_split'
                        , type=float
                        , default=0.2
                        , help="Proportion des données à garder pour le test")
    parser.add_argument('--nrows'
                        , type=int
                        , default=10_000
                        , help="Nombre de lignes à garder pour chaque label")

    args = parser.parse_args()

    # Création du répertoire des données s'il n'existe pas

    # Répertoire des données
    data_dir = args.data_dir

    # Chemin des fichiers
    corpus_path = args.corpus_path
    macrolanguages_path = args.macrolanguages_path

    nrows = args.nrows

    # Création du répertoire des données s'il n'existe pas
    os.makedirs(data_dir, exist_ok=True)

    labels = {'eng': 'English', 'deu': 'German', 'fra': 'French', 'spa': 'Spanish', 'ita': 'Italian',
              'nor': 'Norwegian',
              'tur': 'Turkish',
              'rus': 'Russian',
              'ell': 'Greek', 'dan': 'Danish', 'fin': 'Finnish',
              'ara': 'Arabic', 'heb': 'Hebrew', 'zho': 'Chinese', 'hin': 'Hindi', 'jpn': 'Japanese', 'fas': 'Persian',
              'kor': 'Korean',
              'lat': 'Latin',
              'vie': 'Vietnamese', 'tha': 'Thai'}

    train_data, test_data = preprocess_data(data_dir=data_dir
                                            , corpus_path=corpus_path
                                            , macrolanguages_path=macrolanguages_path
                                            , labels=labels
                                            , test_split=args.test_split
                                            , nrows=nrows)
