import click
import numpy as np
import pandas as pd
from biom import load_table
from skbio.stats.composition import (clr, centralize, closure,
                                     multiplicative_replacement,
                                     _gram_schmidt_basis)

from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import tempfile
from subprocess import Popen
import io
from patsy import dmatrix

import tensorflow as tf
from deep_mae.multimodal import build_model
import pickle


@click.group()
def run_models():
    pass


def load_tables(table1_file, table2_file):
    train_microbes = load_table(table1_file)
    train_metabolites = load_table(table2_file)

    microbes_df = pd.DataFrame(
        np.array(train_microbes.matrix_data.todense()).T,
        index=train_microbes.ids(axis='sample'),
        columns=train_microbes.ids(axis='observation'))

    metabolites_df = pd.DataFrame(
        np.array(train_metabolites.matrix_data.todense()).T,
        index=train_metabolites.ids(axis='sample'),
        columns=train_metabolites.ids(axis='observation'))
    return microbes_df, metabolites_df


@run_models.command()
@click.option('--table1-file',
              help='Input biom table of abundances')
@click.option('--table2-file',
              help='Input metadata file')
@click.option('--output-file',
              help='Saved tensorflow model.')
def run_deep_mae(table1_file, table2_file, output_file):
    lam = 0
    dropout_rate = 0.90
    epochs = 100
    batch_size = 10

    # careful with these parameters
    microbe_latent_dim = 4
    metabolite_latent_dim = 4

    microbes_df, metabolites_df = load_tables(
        table1_file, table2_file)

    # filter out low abundance microbes
    microbe_ids = microbes_df.columns
    metabolite_ids = metabolites_df.columns

    # normalize the microbe and metabolite counts to sum to 1
    microbes = closure(microbes_df)
    metabolites = closure(metabolites_df)
    params = []

    model = build_model(microbes, metabolites,
                        microbe_latent_dim=microbe_latent_dim,
                        metabolite_latent_dim=metabolite_latent_dim,
                        dropout_rate=dropout_rate,
                        lam=lam)
    model.fit(
        {
            'ms_input': metabolites
        },
        {
            'otu_output': microbes,
        },
        verbose=0,
        #callbacks=[tbCallBack],
        epochs=epochs, batch_size=batch_size)

    weights = model.get_weights()
    V1 = weights[0]   # ms input weights
    V1b = weights[1]  # ms input bias
    V2 = weights[2]   # ms shared weights
    V2b = weights[3]  # ms shared bias
    U2 = weights[4]   # otu shared weights
    U2b = weights[5]  # otu shared bias
    U1 = weights[6]  # otu output weights
    U1b = weights[7] # otu output bias

    # print(U1.shape, U2.shape, V2.shape, V1.shape)
    # ranks = - ((((U2.T + U1b) @ U1.T + V2b) @ V2.T + V1b) @ V1.T)
    ranks = - U1.T @ U2.T @ V2.T @ V1.T

    ranks = pd.DataFrame(ranks, index=microbe_ids,
                         columns=metabolite_ids)
    ranks.to_csv(output_file, sep='\t')


@run_models.command()
@click.option('--table1-file',
              help='Input biom table of abundances')
@click.option('--table2-file',
              help='Input metadata file')
@click.option('--output-file',
              help='Saved tensorflow model.')
def run_pearson(table1_file, table2_file, output_file):
    microbes_df, metabolites_df = load_tables(
        table1_file, table2_file)
    n, d1 = microbes_df.shape
    n, d2 = metabolites_df.shape

    pearson_res = np.zeros((d1, d2))
    for i in range(d1):
        for j in range(d2):
            res = pearsonr(microbes_df.iloc[:, i],
                           metabolites_df.iloc[:, j])
            pearson_res[i, j] = res[0]
    ranks = pd.DataFrame(
        pearson_res,
        index=microbes_df.columns,
        columns=metabolites_df.columns)
    ranks.to_csv(output_file, sep='\t')


@run_models.command()
@click.option('--table1-file',
              help='Input biom table of abundances')
@click.option('--table2-file',
              help='Input metadata file')
@click.option('--output-file',
              help='Saved tensorflow model.')
def run_spearman(table1_file, table2_file, output_file):
    microbes_df, metabolites_df = load_tables(
        table1_file, table2_file)
    n, d1 = microbes_df.shape
    n, d2 = metabolites_df.shape

    pearson_res = np.zeros((d1, d2))
    for i in range(d1):
        for j in range(d2):
            res = spearmanr(microbes_df.iloc[:, i],
                           metabolites_df.iloc[:, j])
            pearson_res[i, j] = res[0]
    ranks = pd.DataFrame(
        pearson_res,
        index=microbes_df.columns,
        columns=metabolites_df.columns)
    ranks.to_csv(output_file, sep='\t')


if __name__ == "__main__":
    run_models()
