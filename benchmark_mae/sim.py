import os
import glob
import numpy as np
import pandas as pd
from skbio.stats.composition import ilr_inv
from skbio.stats.composition import clr_inv as softmax


def parse_data(files):
    """ Parses simulation data from multiple files.

    Parameters
    ----------
    file : list of str
        List of file paths

    Returns
    -------
    df : pd.DataFrame
        Dataframe with time, x, y columns along with additional
        columns specifying microbe and chemical abundances.

    Notes
    -----
    This assumes an naming scheme and file structure.
    Each file is organized by x/y coordinates representing
    spatial abundance.  Each filename has the timepoint
    encoded in the name.
    """
    dfs = []
    for f in files:
        fname = os.path.basename(f)
        t = int(fname.split('_')[-1])
        name = '_'.join(fname.split('_')[:-1])
        dat = np.loadtxt(f)
        u, v = dat.shape
        x = np.repeat(np.arange(v).reshape(-1, 1), u, axis=1)
        y = np.repeat(np.arange(u).reshape(-1, 1), v, axis=1)
        df = pd.DataFrame(
            {
                'time': [t] * dat.shape[0] * dat.shape[1],
                'x': x.T.flatten(),
                'y': y.flatten(),
                name: dat.flatten()
            }
        )
        dfs.append(df)
    df = pd.concat(dfs)
    df.index = df.apply(
        lambda x: '%d_%d_%d' % (x['time'], x['x'], x['y']), axis=1)
    return df


def partition_microbes(num_microbes, sigmaQ, microbe_in, state):
    """ Split up a single microbe abundances into multiple strains.

    Parameters
    ----------
    num_microbes : int
        Number of strains to be represented
    sigmaQ : float
        The variance of the multivariate distribution
    microbe_in : np.array
        The input abundances for a single species
    state : numpy random state
        Random number generator

    Returns
    -------
    microbes_out : np.array
        Multiple strain abundances.
    """
    num_samples = len(microbe_in)

    a = state.multivariate_normal(
            mean=np.zeros(num_microbes-1),
            cov=np.diag([sigmaQ] * (num_microbes-1)),
            size=num_samples
    )

    microbe_partition = ilr_inv(a)

    microbes_out = np.multiply(microbe_partition,
                               microbe_in.reshape(-1, 1))
    return microbes_out


def partition_metabolites(uU, sigmaU, uV, sigmaV,
                          num_microbes, num_metabolites,
                          latent_dim, microbe_partition,
                          metabolite_in, state):
    """ Split up a single chemical abundances into multiple subspecies.

    Parameters
    ----------
    uU, sigmaU, uV, sigmaV : int, int, int, int
        Parameters for the conditional probability matrix.
    num_microbes : int
        Number of strains to be represented
    num_metabolites : int
        Number of chemicals to be represented
    latent_dim : int
        Number of latent dimensions in conditional probability
        matrix.
    microbe_partition : np.array
        The input microbial abundances for multiple strains.
    metabolite_in : np.array
        The input intensities for a single chemicals
    state : numpy random state
        Random number generator

    Returns
    -------
    U: np.array
        Microbial latent variables.
    V: np.array
        Metabolomic latent variables.
    metabolites_out: np.array
        Multiple chemical abundances.
    """

    num_samples = len(metabolite_in)

    U = state.normal(
        uU, sigmaU, size=(num_microbes, latent_dim)
    )
    V = state.normal(
        uV, sigmaV, size=(latent_dim, num_metabolites)
    )

    # Randomly generate conditional probability matrices
    # Question : how to incorporate the existing abundances?
    probs = softmax(U @ V)

    # for each submicrobe strain, generate metabolite distribution
    metabolite_partition = microbe_partition @ probs

    # Return partitioned metabolites
    metabolites_out = np.multiply(metabolite_partition,
                                  metabolite_in.reshape(-1, 1))

    return U, V, metabolites_out


def cystic_fibrosis_simulation(data_dir="data"):
    """ Read in cystic fibrosis data"""

    ds = ['F', 'P', 'I', 'SA', 'SG', 'theta_f', 'theta_p']
    df = parse_data(glob.glob(os.path.join(data_dir, "F_*")))
    for s in ds[1:]:
        df_b = parse_data(glob.glob(os.path.join(data_dir, "%s_*" % s)))
        df = pd.merge(df, df_b,
                      left_on=['time', 'x', 'y'],
                      right_on=['time', 'x', 'y'])
    return df
