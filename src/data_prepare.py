import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # data directory
    data_dir = '../data/'

    # create data directory if not exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # download the corpus
    corpus_path = data_dir + 'sentences.tar.bz2'

    # download the macrolanguages
    macrolanguages_path = data_dir + 'iso-639-3-macrolanguages.tab'

    # languages list
    labels = {'eng': 'English', 'deu': 'German', 'fra': 'French', 'spa': 'Spanish', 'ita': 'Italian',
              'nor': 'Norwegian',
              'tur': 'Turkish',
              'rus': 'Russian',
              'ell': 'Greek', 'dan': 'Danish', 'fin': 'Finnish',
              'ara': 'Arabic', 'heb': 'Hebrew', 'zho': 'Chinese', 'hin': 'Hindi', 'jpn': 'Japanese', 'fas': 'Persian',
              'kor': 'Korean',
              'lat': 'Latin',
              'vie': 'Vietnamese', 'tha': 'Thai'}

    # Chargement du corpus
    print(f"Load corpus from file {corpus_path}...")
    data = pd.read_csv(corpus_path, sep='\t',
                       header=None, compression='bz2', low_memory=False)
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

    # remplacer les valeurs NAN dans la colonne 'lang' par la valeur de la colonne 'I_Id'
    data.fillna({'label': data.I_Id}, inplace=True)

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
    nrows = 100_000
    labels_over_nrows = label_counts[label_counts > nrows].index.tolist()

    print(f"labels_over_nrows: {labels_over_nrows}")

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

    print("Limiting to n rows for each label...")

    # Initialiser un DataFrame vide pour le résultat final limité
    data_final = pd.DataFrame()

    # Limiter à 100000 lignes pour chaque label concerné
    for label in labels_over_nrows:

        df_temp = df_final[df_final['label'] == label].sample(n=100000, random_state=42)
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

    print(f"save result in '{data_dir}*_data.pkl'...")

    train_data.to_pickle(data_dir + 'train_data.pkl')
    test_data.to_pickle(data_dir + 'test_data.pkl')

    print(train_data.describe())

    print(f"Preprocessing end...")


if __name__ == "__main__":
    main()
