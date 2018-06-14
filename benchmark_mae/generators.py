import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from skbio.stats.composition import clr_inv as softmax
from biom.util import biom_open
from biom import Table


def deposit(table1, table2, metadata, U, V, B, it, rep, output_dir):
    """ Writes down tables, metadata and feature metadata into files.

    Parameters
    ----------
    table : biom.Table
        Biom table
    metadata : pd.DataFrame
        Dataframe of sample metadata
    feature_metadata : pd.DataFrame
        Dataframe of features metadata
    it : int
        iteration number
    rep : int
        repetition number
    output_dir : str
        output directory
    """
    choice = 'abcdefghijklmnopqrstuvwxyz'
    output_microbes = "%s/table_microbes.%d_%s.biom" % (
        output_dir, it, choice[rep])
    output_metabolites = "%s/table_metabolites.%d_%s.biom" % (
        output_dir, it, choice[rep])
    output_md = "%s/metadata.%d_%s.txt" % (
        output_dir, it, choice[rep])
    output_U = "%s/U.%d_%s.txt" % (
        output_dir, it, choice[rep])
    output_V = "%s/V.%d_%s.txt" % (
        output_dir, it, choice[rep])
    output_B = "%s/B.%d_%s.txt" % (
        output_dir, it, choice[rep])
    output_ranks = "%s/ranks.%d_%s.txt" % (
        output_dir, it, choice[rep])

    idx1 = table1.sum(axis=0) > 0
    idx2 = table2.sum(axis=0) > 0
    table1 = table1.loc[:, idx1]
    table2 = table2.loc[:, idx2]

    table1 = Table(table1.values.T, table1.columns, table1.index)
    table2 = Table(table2.values.T, table2.columns, table2.index)

    with biom_open(output_microbes, 'w') as f:
        table1.to_hdf5(f, generated_by='moi1')
    with biom_open(output_metabolites, 'w') as f:
        table2.to_hdf5(f, generated_by='moi2')

    ranks = (U @ V)

    ranks = ranks[idx1, :]
    ranks = ranks[:, idx2]
    ranks = pd.DataFrame(ranks, index=table1.ids(axis='observation'),
                         columns=table2.ids(axis='observation'))
    ranks.to_csv(output_ranks, sep='\t')
    metadata.to_csv(output_md, sep='\t', index_label='#SampleID')

    B = B[:, idx1]

    np.savetxt(output_U, U)
    np.savetxt(output_V, V)
    np.savetxt(output_B, B)


def random_sigmoid_multimodal(
        num_microbes=20, num_metabolites=100, num_samples=100,
        num_latent_microbes=5, num_latent_metabolites=10,
        num_latent_shared=3, low=-1, high=1,
        microbe_total=10, metabolite_total=100,
        uB=0, sigmaB=2, sigmaQ=0.1,
        uU1=0, sigmaU1=1, uU2=0, sigmaU2=1,
        uV1=0, sigmaV1=1, uV2=0, sigmaV2=1,
        seed=0):
    """
    Parameters
    ----------
    num_microbes : int
       Number of microbial species to simulate
    num_metabolites : int
       Number of molecules to simulate
    num_samples : int
       Number of samples to generate
    num_latent_microbes :
       Number of latent microbial dimensions
    num_latent_metabolites
       Number of latent metabolite dimensions
    num_latent_shared
       Number of dimensions in shared representation
    low : float
       Lower bound of gradient
    high : float
       Upper bound of gradient
    microbe_total : int
       Total number of microbial species
    metabolite_total : int
       Total number of metabolite species
    uB : float
       Mean of regression coefficient distribution
    sigmaB : float
       Standard deviation of regression coefficient distribution
    sigmaQ : float
       Standard deviation of error distribution
    uU1 : float
       Mean of microbial input projection coefficient distribution
    sigmaU1 : float
       Standard deviation of microbial input projection
       coefficient distribution
    uU2 : float
       Mean of microbe output projection coefficient distribution
    sigmaU2 : float
       Standard deviation of microbe output projection
       coefficient distribution
    uV1 : float
       Mean of metabolite input projection coefficient distribution
    sigmaU1 : float
       Standard deviation of metabolite input projection
       coefficient distribution
    uV2 : float
       Mean of metabolite output projection coefficient distribution
    sigmaU2 : float
       Standard deviation of metabolite output projection
       coefficient distribution
    seed : float
       Random seed
    Returns
    -------
    microbe_counts : pd.DataFrame
       Count table of microbial counts
    metabolite_counts : pd.DataFrame
       Count table of metabolite counts
    """
    k = num_latent_shared
    state = check_random_state(seed)
    # only have two coefficients
    beta = state.normal(uB, sigmaB, size=(2, k))

    X = np.vstack((np.ones(num_samples),
                   np.linspace(low, high, num_samples))).T

    Q = np.tanh(state.normal(X @ beta, sigmaQ))

    U1 = state.normal(
        uU1, sigmaU1, size=(num_latent_microbes, num_microbes))
    U2 = state.normal(
        uU2, sigmaU2, size=(k, num_latent_microbes))
    V1 = state.normal(
        uV1, sigmaV1, size=(num_latent_metabolites, num_metabolites))
    V2 = state.normal(
        uV2, sigmaV2, size=(k, num_latent_metabolites))

    def multinomial(n, p):
        return np.vstack([np.random.multinomial(n, p[i, :])
                          for i in range(p.shape[0])]).T

    microbe_counts = multinomial(microbe_total, softmax((Q @ U2 @ U1).T))
    metabolite_counts = multinomial(metabolite_total, softmax((Q @ V2 @ V1).T))
    otu_ids = ['OTU_%d' % d for d in range(microbe_counts.shape[1])]
    ms_ids = ['metabolite_%d' % d for d in range(metabolite_counts.shape[1])]
    sample_ids = ['sample_%d' % d for d in range(metabolite_counts.shape[0])]

    microbe_counts = pd.DataFrame(
        microbe_counts, index=sample_ids, columns=otu_ids)
    metabolite_counts = pd.DataFrame(
        metabolite_counts, index=sample_ids, columns=ms_ids)

    return microbe_counts, metabolite_counts, X, Q, U1, U2, V1, V2


def random_multimodal(num_microbes=20, num_metabolites=100, num_samples=100,
                      latent_dim=3, low=-1, high=1,
                      microbe_total=10, metabolite_total=100,
                      uB=0, sigmaB=2, sigmaQ=0.1,
                      uU=0, sigmaU=1, uV=0, sigmaV=1,
                      seed=0):
    """
    Parameters
    ----------
    num_microbes : int
       Number of microbial species to simulate
    num_metabolites : int
       Number of molecules to simulate
    num_samples : int
       Number of samples to generate
    latent_dim :
       Number of latent dimensions
    low : float
       Lower bound of gradient
    high : float
       Upper bound of gradient
    microbe_total : int
       Total number of microbial species
    metabolite_total : int
       Total number of metabolite species
    uB : float
       Mean of regression coefficient distribution
    sigmaB : float
       Standard deviation of regression coefficient distribution
    sigmaQ : float
       Standard deviation of error distribution
    uU : float
       Mean of microbial input projection coefficient distribution
    sigmaU : float
       Standard deviation of microbial input projection
       coefficient distribution
    uV : float
       Mean of metabolite output projection coefficient distribution
    sigmaV : float
       Standard deviation of metabolite output projection
       coefficient distribution
    seed : float
       Random seed

    Returns
    -------
    microbe_counts : pd.DataFrame
       Count table of microbial counts
    metabolite_counts : pd.DataFrame
       Count table of metabolite counts
    """
    state = check_random_state(seed)
    # only have two coefficients
    beta = state.normal(uB, sigmaB, size=(2, num_microbes))

    X = np.vstack((np.ones(num_samples),
                   np.linspace(low, high, num_samples))).T

    microbes = softmax(state.normal(X @ beta, sigmaQ))

    U = state.normal(
        uU, sigmaU, size=(num_microbes, latent_dim))
    V = state.normal(
        uV, sigmaV, size=(latent_dim, num_metabolites))

    probs = softmax(U @ V)
    microbe_counts = np.zeros((num_samples, num_microbes))
    metabolite_counts = np.zeros((num_samples, num_metabolites))
    n1 = microbe_total
    n2 = metabolite_total // microbe_total
    for n in range(num_samples):
        otu = np.random.multinomial(n1, microbes[n, :])
        for i in range(num_microbes):
            ms = np.random.multinomial(otu[i] * n2, probs[i, :])
            metabolite_counts[n, :] += ms
        microbe_counts[n, :] += otu

    otu_ids = ['OTU_%d' % d for d in range(microbe_counts.shape[1])]
    ms_ids = ['metabolite_%d' % d for d in range(metabolite_counts.shape[1])]
    sample_ids = ['sample_%d' % d for d in range(metabolite_counts.shape[0])]

    microbe_counts = pd.DataFrame(
        microbe_counts, index=sample_ids, columns=otu_ids)
    metabolite_counts = pd.DataFrame(
        metabolite_counts, index=sample_ids, columns=ms_ids)

    # if np.any(microbe_counts.sum(axis=1) == 0):
    #     raise ValueError('Unobserved OTUs')
    # if np.any(metabolites_counts.sum(axis=1) == 0):
    #     raise ValueError('Unobserved metabolites')

    return microbe_counts, metabolite_counts, X, beta, U, V
