import csv
import numpy as np
import pandas as pd
from random import randint
import matplotlib.pyplot as plt


pd.set_option('display.max_rows', 200)

dataset = r"dataset_paris.csv"


def modif(file):
    #on  lit le dataset brut
    data = pd.read_csv(file, sep=';', low_memory=False)

    #on récupère les colonnes
    columns_used = ["Date et heure de comptage","Taux d'occupation"]

    #on retire toutes les colonnes dont on ne va pas se servir
    to_drop = data.columns.difference(columns_used)
    data.drop(to_drop, axis = 1, inplace=True)


    #on trie par date et heure de comptage
    data.sort_values([columns_used[0]], inplace=True)
    data.reset_index(drop=True, inplace=True)

    #on retire les lignes inutiles et on renvoie le tableau sans valeurs manquantes
    data = data[data[columns_used[1]].notna()]

    #on regroupe les lignes par même jour même heure et on fait la moyenne
    test = data.groupby([columns_used[0]]).mean()

    data.reset_index(drop=True, inplace=True)


    matrice = data.to_numpy()

    n = len(test)

    matrice_zeros = np.zeros((n//24 +1,24))

    for i in range(0,n//24):
        date = matrice[i][0]
        for j in range(24):
            matrice_zeros[i][j] = matrice[i*24 + j][1]

    matrice_zeros = np.delete(matrice_zeros,(n//24), axis=0)


    return matrice_zeros



def recup_data_avenue(file):
    # on  lit le dataset brut
    data = pd.read_csv(file, sep=';', low_memory=False)

    # on récupère les colonnes
    columns_used = ["Date et heure de comptage", "Taux d'occupation"]

    data = data[data["Libelle"] == "La_Fayette"]


    # on retire toutes les colonnes dont on ne va pas se servir
    to_drop = data.columns.difference(columns_used)
    data.drop(to_drop, axis=1, inplace=True)

    # on trie par date et heure de comptage
    data.sort_values([columns_used[0]], inplace=True)
    data.reset_index(drop=True, inplace=True)

    #on retire les lignes inutiles et on renvoie le tableau sans valeurs manquantes
    data = data[data[columns_used[1]].notna()]


    #on regroupe les lignes par même jour même heure et on fait la moyenne
    test = data.groupby([columns_used[0]]).mean()


    data.reset_index(drop=True, inplace=True)

    #on transforme les données panda en données numpy
    matrice = data.to_numpy()

    n = len(test)

    matrice_zeros = np.zeros((n//24 +1,24))

    for i in range(0,n//24):
        date = matrice[i][0]
        for j in range(24):
            matrice_zeros[i][j] = matrice[i*24 + j][1]

    matrice_zeros = np.delete(matrice_zeros,(n//24), axis=0)


    return matrice_zeros


