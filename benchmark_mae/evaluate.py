from biom import load_table
import pandas as pd
from os.path import basename, splitext
import numpy as np
from scipy.stats import spearmanr


def top_absolute_results(result_files, truth_files, output_file, top_N=10):
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
    output_file : str
        File path for confusion matrix summary.

    Note
    ----
    This assumes that the tables all have the same basename.
    """
    # only use the result files that match with the output_file
    out_suf = splitext(basename(output_file))[0]

    result_files = list(filter(lambda x: out_suf in basename(x), result_files))
    index_names = list(map(lambda x: splitext(basename(x))[1], result_files))
    col_names = ['%s_TP' % out_suf,
                 '%s_FP' % out_suf,
                 '%s_FN' % out_suf,
                 '%s_TN' % out_suf,
                 '%s_RK' % out_suf,
    ]
    TP, FP, FN, TN, RC = 0, 1, 2, 3, 4

    stats = pd.DataFrame(columns=col_names, index=index_names)
    for r_file, t_file in zip(result_files, truth_files):
        res = pd.read_table(r_file, sep='\t', index_col=0)
        exp = pd.read_table(t_file, sep='\t', index_col=0)

        res = pd.melt(res, id_vars=['index'],
                      var_name='metabolite', value_name='rank')
        res.rename(columns={'index': 'OTU'})

        exp = pd.melt(exp, id_vars=['index'],
                      var_name='metabolite', value_name='rank')
        exp.rename(columns={'index': 'OTU'})

        exp = exp.sort_values(by='rank', ascending=False)
        res = res.sort_values(by='rank', ascending=False)
        ids = exp.index[:top_N]
        rank_stat = spearmanr(res.loc[ids, 'rank'].values,
                              exp.loc[ids, 'rank'].values).correlation
        ids = set(exp.index)
        hits = set(res.iloc[:top_N].index)
        truth = set(exp.iloc[:top_N].index)

        x = pd.Series(
            {col_names[TP]: len(hits & truth),
             col_names[FP]: len(hits - truth),
             col_names[FN]: len(truth - hits),
             col_names[TN]: len((ids-hits) & (ids-truth)),
             col_names[RC]: rank_stat
            })
        stats.loc[tab_file] = x
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

    Note
    ----
    This assumes that table_files and metadata_files
    are in the same matching order.

    table.xxx.biom
    metadata.xxx.txt
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
