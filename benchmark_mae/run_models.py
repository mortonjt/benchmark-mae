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
from deep_mae.multimodal import build_model, onehot
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
    epochs = 10
    batch_size = 100

    # careful with these parameters
    microbe_latent_dim = 3
    metabolite_latent_dim = 3

    microbes_df, metabolites_df = load_tables(
        table1_file, table2_file)

    # filter out low abundance microbes
    microbe_ids = microbes_df.columns
    metabolite_ids = metabolites_df.columns

    # normalize the microbe and metabolite counts to sum to 1
    #microbes = closure(microbes_df)
    #metabolites = closure(metabolites_df)
    otu_hits, ms_hits, sample_ids = onehot(
        microbes_df.values, closure(metabolites_df.values))
    params = []

    # model = build_model(microbes, metabolites,
    #                     microbe_latent_dim=microbe_latent_dim,
    #                     metabolite_latent_dim=metabolite_latent_dim,
    #                     dropout_rate=dropout_rate,
    #                     lam=lam)
    model = build_model(
        microbes_df, metabolites_df,
        latent_dim=microbe_latent_dim, lam=lam,
        dropout_rate=0.9
    )

    model.fit(
        {
            'otu_input': otu_hits,
        },
        {
            'ms_output': ms_hits
        },
        verbose=0,
        #callbacks=[tbCallBack],
        epochs=epochs, batch_size=batch_size)

    weights = model.get_weights()
    U, V = weights[0], weights[1]
    ranks = U @ V
    ranks = pd.DataFrame(ranks, index=microbe_ids,
                         columns=metabolite_ids)
    ranks.to_csv(output_file, sep='\t')


def optimize(log_loss, learning_rate, beta1, beta2, clipping_size):
    """ Perform optimization (via Gradient Descent)"""
    with tf.name_scope('optimize'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(log_loss))
        gradients, _ = tf.clip_by_global_norm(gradients, clipping_size)

        train_ = optimizer.apply_gradients(zip(gradients, variables))
        return train_, gradients, variables

def build_tf_model(d1, d2, u_mean=0, u_scale=1, v_mean=0, v_scale=1,
                   batch_size=50, latent_dim=3, dropout_rate=0.5, lam=0,
                   learning_rate = 0.1, beta_1=0.999, beta_2=0.9999, clipnorm=10.):
    """ Build a tensorflow model

    Parameters
    ----------
    microbes : np.array
       One hot encodings of microbes
    metabolties : np.array
       Table of metabolite abundances

    Returns
    -------
    loss : tf.Tensor
       The log loss of the model.
    """
    p = latent_dim

    # TODO: Make sure that these shapes are comparable
    total_count = tf.placeholder(tf.float32, [batch_size], name='total_count')
    Y_ph = tf.placeholder(tf.float32, [batch_size, d2], name='Y_ph')
    X_ph = tf.placeholder(tf.int32, [batch_size], name='X_ph')
    sample_ph = tf.placeholder(tf.float32, [batch_size], name='samples')

    qU = tf.Variable(tf.random_normal([d1, p]), name='qU')
    qV = tf.Variable(tf.random_normal([p, d2]), name='qV')

    # regression coefficents distribution
    U = Normal(loc=tf.zeros([d1, p]) + u_mean,
               scale=tf.ones([d1, p]) * u_scale,
               name='U')
    V = Normal(loc=tf.zeros([p, d2]) + v_mean,
               scale=tf.ones([p, d2]) * v_scale,
               name='V')

    du = tf.gather(qU, X_ph, axis=0)
    dv = du @ qV
    Y = Multinomial(total_count=total_count, logits=dv)
    log_loss = - (
        tf.reduce_sum(Y.log_prob(Y_ph)) * (num_samples / batch_size) + \
        tf.reduce_sum(U.log_prob(qU)) + tf.reduce_sum(V.log_prob(qV)))
    train, grad, var = optimize(log_loss, learning_rate,
                                beta_1, beta_2, clipnorm)
    return train, grad, var, log_loss, qU, qV, X_ph, Y_ph, total_count


@run_models.command()
@click.option('--table1-file',
              help='Input biom table of abundances')
@click.option('--table2-file',
              help='Input metadata file')
@click.option('--output-file',
              help='Saved tensorflow model.')
def run_tf_mae(table1_file, table2_file, output_file):
    lam = 0

    epochs = 10
    batch_size = 100
    learning_rate = 0.1
    u_mean, u_scale = 0, 1
    v_mean, v_scale = 0, 1

    microbes_df, metabolites_df = load_tables(
        table1_file, table2_file)

    # filter out low abundance microbes
    microbe_ids = microbes_df.columns
    metabolite_ids = metabolites_df.columns

    # normalize the microbe and metabolite counts to sum to 1
    #microbes = closure(microbes_df)
    #metabolites = closure(metabolites_df)
    otu_hits, ms_hits, sample_ids = onehot(
        microbes_df.values, closure(metabolites_df.values))
    params = []

    otu_hits, ms_hits, sample_ids = onehot(microbes.values, metabolites.values)

    with tf.Graph().as_default(), tf.Session() as session:
        d1, d2 = microbes.shape[1], metabolites.shape[1]

        res = build_model(
            d1, d2, latent_dim=latent_dim, batch_size=batch_size,
            u_mean=u_mean, u_scale=u_scale, v_mean=v_mean, v_scale=v_scale,
            learning_rate=learning_rate, beta_1=0.9, beta_2=0.99, clipnorm=10.)
        train, grad, var, log_loss, qU, qV, X_ph, Y_ph, total_count = res
        tf.global_variables_initializer().run()
        losses = np.array([0] * num_iter, dtype=np.float32)
        err = np.array([0] * num_iter, dtype=np.float32)
        for i in range(0, num_iter):
            batch = np.random.randint(
                otu_hits.shape[0], size=batch_size)
            batch_ids = sample_ids[batch]
            total = metabolites.values[batch_ids, :].sum(axis=1).astype(np.float32)
            train_, loss, rU, rV = session.run(
                [train, log_loss, qU, qV],
                feed_dict={
                    X_ph: otu_hits[batch],
                    Y_ph: metabolites.values[batch_ids, :],
                    total_count: total
                }
            )

    ranks = rU @ rV
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
