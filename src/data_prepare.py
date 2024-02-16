import pandas as pd

import os


def main():
    # data directory
    data_dir = '../data'

    # create data directory if not exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # download the corpus
    corpus_path = data_dir + '/sentences.tar.bz2'

    # download the macrolanguages
    macrolanguages_path = data_dir + '/iso-639-3-macrolanguages.tab'

    # languages list
    langs = {'eng', 'pol', 'deu', 'fra', 'spa', 'ita',
             'tur',
             'por', 'rus', 'ukr', 'nld', 'bul',
             'ell', 'swe',
             'hun', 'gle', 'lav', 'dan', 'fin',
             'ara', 'heb', 'zho', 'hin', 'jpn', 'fas',
             'kor',
             'hye', 'swa', 'ber', 'ces', 'lat',
             'nor', 'ron', 'slk', 'hbs',
             'mkd',
             'vie', 'est', 'tha'}

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
    macro_lang.columns = ['lang', 'I_Id', 'I_Status']

    print(f"Preprocessing start...")

    # jointure entre la table des sentences et la table macrolanguages
    data = pd.merge(data, macro_lang, how='left', on='I_Id')

    # remplacer les valeurs NAN dans la colonne 'lang' par la valeur de la colonne 'I_Id'
    data.fillna({'lang': data.I_Id}, inplace=True)

    # suppression des columns 'I_Status', 'id' et 'I_Id'
    data = data.drop(columns=['I_Status', 'id', 'I_Id'])

    # garder que les langues selectionnées dans la liste langs
    data = data[data['lang'].isin(langs)]

    print(f"save result in '{data_dir}data.pkl'...")

    data.to_pickle(data_dir + 'data.pkl')

    print(data.describe())

    print(f"Preprocessing end...")


if __name__ == "__main__":
    main()
