from biom import load_table
import pandas as pd
from os.path import basename, splitext
import numpy as np
from scipy.stats import spearmanr
from skbio.stats.composition import clr_inv
from maestro.util import rank_hits


def rank_accuracy(res, exp, top_N):
    ids = exp.index[:top_N]
    rank_stats = []
    tps = fps = fns = tns = 0
    ids = set(exp.columns)
    for i in exp.index:
        exp_idx = np.argsort(exp.loc[i, :].values)
        res_idx = np.argsort(res.loc[i, :].values)
        exp_names = exp.columns[exp_idx[-top_N:]]
        res_names = res.columns[res_idx[-top_N:]]

        r = spearmanr(exp.loc[i, exp_names],
                      res.loc[i, exp_names])

        hits  = set(res_names)
        truth = set(exp_names)

        tps += len(hits & truth)
        fns += len(truth - hits)
        fps += len(hits - truth)
        tns += len((ids - hits) & (ids - truth))
        rank_stats.append(r.correlation)

    return rank_stats, tps, fps, fns, tns

def top_rank_accuracy(res, exp, B, top_OTU, top_MS):
    lo = np.argsort(B[1, :])[:top_OTU]
    hi = np.argsort(B[1, :])[-top_OTU:]
    tps = fps = fns = tns = 0
    ids = set(exp.columns)
    lo_r = []

    for i in lo:

        ridx = np.argsort(exp.iloc[i, :]).values[-top_MS:]
        eidx = np.argsort(res.iloc[i, :]).values[-top_MS:]
        exp_names = exp.columns[eidx]
        res_names = res.columns[ridx]
        hits  = set(res_names)
        truth = set(exp_names)
        tps += len(hits & truth)
        fns += len(truth - hits)
        fps += len(hits - truth)
        tns += len((ids - hits) & (ids - truth))

        r = spearmanr(res.iloc[i, eidx], exp.iloc[i, eidx])
        lo_r.append(r.correlation)

    hi_r = []
    for i in hi:
        ridx = np.argsort(exp.iloc[i, :]).values[-top_MS:]
        eidx = np.argsort(res.iloc[i, :]).values[-top_MS:]
        exp_names = exp.columns[eidx]
        res_names = res.columns[ridx]
        hits  = set(res_names)
        truth = set(exp_names)

        tps += len(hits & truth)
        fns += len(truth - hits)
        fps += len(hits - truth)
        tns += len((ids - hits) & (ids - truth))

        r = spearmanr(res.iloc[i, ridx], exp.iloc[i, eidx])
        hi_r.append(r.correlation)
    rank_stats = lo_r + hi_r

    return rank_stats, tps, fps, fns, tns


def top_absolute_results(result_files, truth_files, parameter_files,
                         output_file, top_OTU=10, top_MS=20):
    """ Computes confusion matrice over all runs for a
    specified set of results for top results.

    This only looks for absolute change. So this is agnostic of which
    sample class it belongs to.

    Parameters
    ----------
    result_files : list of str
        List of filepaths for estimated correlated features.
    truth_files : list of str
        List of filepaths for ground truth correlated features.
    parameter_files : list of str
        List of filepaths for ground truth simulation parameters.
        These are the regression coefficients used to generate the
        microbial table.
    output_file : str
        File path for confusion matrix summary.

    Note
    ----
    This assumes that the tables all have the same basename.
    """
    # only use the result files that match with the output_file
    out_suf = splitext(basename(output_file))[0]

    result_files = list(filter(lambda x: out_suf in basename(x), result_files))
    index_names = list(map(lambda x: splitext(basename(x))[0], result_files))
    col_names = ['%s_TP' % out_suf,
                 '%s_FP' % out_suf,
                 '%s_FN' % out_suf,
                 '%s_TN' % out_suf,
                 '%s_meanRK' % out_suf]
    TP, FP, FN, TN, meanRK = 0, 1, 2, 3, 4
    x = pd.Series(index=col_names)
    stats = {}
    for name, r_file, t_file, p_file in zip(
            index_names, result_files, truth_files, parameter_files):
        res = pd.read_table(r_file, sep='\t', index_col=0)
        exp = pd.read_table(t_file, sep='\t', index_col=0)
        B = np.loadtxt(p_file)
        rank_stats, tps, fps, fns, tns = top_rank_accuracy(
            res, exp, B, top_OTU, top_MS)

        # rank_stats, tps, fps, fns, tns = rank_accuracy(
        #     res, exp, top_MS)

        x = pd.Series({
            col_names[TP]: tps,
            col_names[FP]: fps,
            col_names[FN]: fns,
            col_names[TN]: tns,
            col_names[meanRK]: np.mean(rank_stats)
        })
        stats[name] = x

    stats = pd.DataFrame(stats, index=col_names).T
    stats.to_csv(output_file, sep='\t')


def aggregate_summaries(confusion_matrix_files, metadata_files,
                        axis, output_file):
    """ Aggregates summary files together, along with the variable of interest.

    Parameters
    ----------
    confusion_matrix_files : list of str
        List of filepaths for summaries.
    metadata_files : list of str
        List of filepaths for metadata files.
    axis : str
        Category of differentiation.
    output_file : str
        Output path for aggregated summaries.

    """
    mats = [pd.read_table(f, index_col=0) for f in confusion_matrix_files]
    merged_stats = pd.concat(mats, axis=1)

    # first
    index_names = list(map(lambda x: splitext(basename(x))[1], metadata_files))

    # aggregate stats in all metadata_files. For now, just take the mean
    x = [pd.read_table(f, index_col=0)[axis].mean() for f in metadata_files]
    cats = pd.DataFrame(x, index=index_names, columns=[axis])
    merged_stats = pd.merge(merged_stats, cats, left_index=True, right_index=True)
    merged_stats.to_csv(output_file, sep='\t')


def edge_roc_curve(ranks_file, edges_file, output_file, k_max = None):

    ranks = pd.read_table(ranks_file, index_col=0)
    ranks = ranks.apply(clr_inv, axis=1)
    if k_max is None:
        k_max = min(ranks.shape)
    edges = pd.read_table(edges_file, index_col=0)
    pos, neg =  _edge_roc_curve(ranks, edges, k_max)
    df = pd.merge(pos, neg, left_index=True, right_index=True,
                  suffixes=('pos', 'neg'))
    df.to_csv(output_file, sep='\t')


def _edge_roc_curve(ranks, edges, k_max=10):
    """ Create a ROC curve based on nearest neighbors.

    Parameters
    ----------
    ranks : pd.DataFrame
        Estimate microbe-metabolite ranks.
    edges : pd.DataFrame
        List of ground truth edges. Note that this data frame
        must have the columns 'microbe', 'metabolite', 'direction'.
    k_high : int
        Maximum number of nearest neighbors to evaluate.

    Returns
    -------
    pos_results : pd.DataFrame
        TP : np.array
            List of True positives for each k
        TN : np.array
            List of True negatives for each k
        FP : np.array
            List of False positives for each k
        FN : np.array
            List of False negatives for each k
    neg_results : pd.DataFrame
        TP : np.array
            List of True positives for each k
        TN : np.array
            List of True negatives for each k
        FP : np.array
            List of False positives for each k
        FN : np.array
            List of False negatives for each k
    """
    all_edges = set(map(tuple, edges[['microbe', 'metabolite']].values))

    pos_results = []
    neg_results = []

    idx = np.arange(0, k_max+1)
    for k in idx:
        res_pos_edges = rank_hits(ranks.T, k, pos=True)
        res_neg_edges = rank_hits(ranks.T, k, pos=False)

        exp_pos_edges = edges.loc[edges.direction==1]
        exp_neg_edges = edges.loc[edges.direction==-1]

        exp_pos_edges = set(
            map(tuple, exp_pos_edges[['metabolite', 'microbe']].values))
        exp_neg_edges = set(
            map(tuple, exp_neg_edges[['metabolite', 'microbe']].values))

        res_pos_edges = set(
            map(tuple, res_pos_edges[['src', 'dest']].values))
        res_neg_edges = set(
            map(tuple, res_neg_edges[['src', 'dest']].values))

        # take an intersection of all of the positive edges
        TP = len(exp_pos_edges & res_pos_edges)
        TN = len((all_edges - exp_pos_edges) & (all_edges - res_pos_edges))
        FP = len(res_pos_edges - exp_pos_edges)
        FN = len(exp_pos_edges - res_pos_edges)
        pos_results.append({'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN})

        # take an intersection of all of the negative edges
        TP = len(exp_neg_edges & res_neg_edges)
        TN = len((all_edges - exp_pos_edges) & (all_edges - res_pos_edges))
        FP = len(res_neg_edges - exp_neg_edges)
        FN = len(exp_neg_edges - res_neg_edges)
        neg_results.append({'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN})

    pos_results = pd.DataFrame(pos_results, index=idx)
    neg_results = pd.DataFrame(neg_results, index=idx)
    return pos_results, neg_results




