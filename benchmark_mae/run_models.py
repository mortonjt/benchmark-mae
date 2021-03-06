#!/usr/bin/env python
import click
import numpy as np
import pandas as pd
from biom import load_table
from skbio.stats.composition import (clr, centralize, closure,
                                     multiplicative_replacement,
                                     _gram_schmidt_basis)

from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import ttest_ind, mannwhitneyu, pearsonr, spearmanr
from scipy.sparse import coo_matrix
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import tempfile
from subprocess import Popen
import io
from patsy import dmatrix
from skbio.stats.composition import clr, centralize, clr_inv
from skbio.stats.composition import clr_inv as softmax
import tensorflow as tf
from tensorflow.contrib.distributions import Multinomial, Normal
# note that the name will change
from rhapsody.multimodal import MMvec
from songbird.multinomial import MultRegression

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

    microbes_df, metabolites_df = load_tables(
        table1_file, table2_file)
    d1, d2 = microbes_df.shape[1], metabolites_df.shape[1]
    num_samples = microbes_df.shape[0]

    # parameters
    epochs = 100000
    batch_size = 5000
    learning_rate = 1e-3
    u_mean, u_scale = 0, 1
    v_mean, v_scale = 0, 1
    latent_dim = 1

    # filter out low abundance microbes
    microbe_ids = microbes_df.columns
    metabolite_ids = metabolites_df.columns

    # normalize the microbe and metabolite counts to sum to 1
    #microbes = closure(microbes_df)
    #metabolites = closure(metabolites_df)
    params = []

    with tf.Graph().as_default(), tf.Session() as session:
        model = MMvec(u_mean=0, u_scale=1, v_mean=0, v_scale=1,
                      batch_size=batch_size, latent_dim=latent_dim,
                      learning_rate=learning_rate, beta_1=0.85, beta_2=0.9,
                      clipnorm=10., save_path=None)
        model(session, coo_matrix(microbes_df.values), metabolites_df.values)
        model.fit(epoch=epochs)

        U, V = model.U, model.V
        d1 = U.shape[0]

        U_ = np.hstack(
            (np.ones((model.U.shape[0], 1)),
             model.Ubias.reshape(-1, 1), U)
        )
        V_ = np.vstack(
            (model.Vbias.reshape(1, -1),
             np.ones((1, model.V.shape[1])), V)
        )

        ranks = pd.DataFrame(
            np.log(softmax(np.hstack(
                (np.zeros((model.U.shape[0], 1)), U_ @ V_)))),
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


@run_models.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def run_ttest(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    cats = metadata[category]
    cs = np.unique(cats)
    def func(x):
        return ttest_ind(*[x[cats == k] for k in cs])
    m, p = np.apply_along_axis(func, axis=0,
                               arr=table.values)
    res = pd.DataFrame({'statistic': m, 'pvalue': p}, index=table.columns)
    res.to_csv(output_file, sep='\t')


@run_models.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def run_mannwhitney(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    cats = metadata[category]
    cs = np.unique(cats)
    def func(x):
        try: # catches the scenario where all values are the same.
            return mannwhitneyu(*[x[cats == k] for k in cs])
        except:
            return 0, 1

    m, p = np.apply_along_axis(func, axis=0,
                               arr=table.values)
    res = pd.DataFrame({'statistic': m, 'pvalue': p}, index=table.columns)
    res.to_csv(output_file, sep='\t')


@run_models.command()
@click.option('--table-file',
              help='Input biom table of abundances')
@click.option('--metadata-file',
              help='Input metadata file')
@click.option('--category',
              help='Category specifying groups')
@click.option('--output-file',
              help='output file of differientially abundance features.')
def run_multinomial(table_file, metadata_file, category, output_file):
    metadata = pd.read_table(metadata_file, index_col=0)
    table = load_table(table_file)
    table = pd.DataFrame(np.array(table.matrix_data.todense()).T,
                         index=table.ids(axis='sample'),
                         columns=table.ids(axis='observation'))
    model = MultRegression(
        batch_size=3, learning_rate=1e-3, beta_scale=1)
    Y = table.values
    X = metadata[['intercept', category]].values
    trainX = X[:-5]
    trainY = Y[:-5]
    testX = X[-5:]
    testY = Y[-5:]
    with tf.Graph().as_default(), tf.Session() as session:
        model(session, trainX, trainY, testX, testY)
        loss, cv, _ = model.fit(epoch=int(1000))
        beta_ = clr(
            clr_inv(
                np.hstack((np.zeros((model.p, 1)), model.B))
            )
        )
    res = pd.DataFrame(
        beta_.T, columns=['intercept', 'statistic'],
        index=table.columns
    )
    res.to_csv(output_file, sep='\t')


if __name__ == "__main__":
    run_models()
