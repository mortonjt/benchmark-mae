import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from biom.util import biom_open
from benchmark_mae.generators import (
    deposit, random_biofilm
)
import yaml


# snakemake config
iteration = 12
config_file = 'params%d.yaml' % iteration
workflow_type = 'local'
local_cores = 1
cores = 4
jobs = 1
force = True
snakefile = 'Snakefile'
dry_run = False
output_dir = 'test_benchmark%d/' % iteration
quiet=False
keep_logger=True
cluster_config = 'cluster.json'
latency_wait = 120
profile = os.path.dirname(__file__)
restart_times = 1

# simulation parameters
regenerate_simulations = False

num_metabolites = 50
num_microbes = 20
num_samples = 126

uU = 0
sigmaU = 0.5
uV = 0
sigmaV = 0.5
latent_dim = 3
sigmaQ = 0.1

microbe_total = 5e2
metabolite_total = 10e8

microbe_kappa = 2.5
metabolite_kappa = 1

timepoint = 9
seed = None

# benchmark parameters
top_OTU = 20      # top OTUs to evaluate
top_MS = 20       # top metabolites to evaluate

timepoint = 9
intervals = 3
benchmark = 'effect_size'
reps = 1
tools = ['deep_mae', 'pearson', 'spearman']

sample_ids = []
if regenerate_simulations:

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    for r in range(reps):
        sample_id = i
        ef = np.round(ef, decimals=2)
        res = random_biofilm(
            df, uU=uU, sigmaU=sigmaU, uV=uV, sigmaV=sigmaV,
            sigmaQ=sigmaQ, latent_dim=latent_dim,
            num_microbes=num_microbes,
            num_metabolites=num_metabolites,
            microbe_total=microbe_total,
            microbe_kappa=microbe_kappa,
            metabolite_total=metabolite_total,
            metabolite_kappa=metabolite_kappa,
            timepoint=timepoint, seed=seed)
        edges, U, V, microbe_counts, metabolite_counts = res

        # setup metadata
        X = pd.DataFrame(X, index=microbe_counts.index,
                         columns=['X%d' % d for d in range(X.shape[1])])
        metadata = X
        metadata['effect_size'] = ef
        deposit(microbe_counts, metabolite_counts,
                metadata, U, V, edges,
                sample_id, r, output_dir)
        sample_ids.append(sample_id)

    # generate config file
    data = {'benchmark': benchmark,
            'intervals': intervals,
            'output_dir': output_dir,
            'samples': sample_ids,
            'reps': reps,
            'tools': tools,
            'top_MS': top_MS,
            'top_OTU': top_OTU,
            # parameters to simulate the model
            'num_samples' : num_samples,
            'num_microbes' : num_microbes,
            'num_metabolites' : num_metabolites,
            'microbe_total' : microbe_total,
            'metabolite_total' : metabolite_total,
            'latent_dim' : latent_dim,
            'sigmaB' : sigmaB,
            'sigmaQ' : sigmaQ,
            'uU' : uU,
            'sigmaU' : sigmaU,
            'uV' : uV,
            'sigmaV' : sigmaV,
            'low' : low,
            'high' : high,
    }
    with open(config_file, 'w') as yfile:
        yaml.dump(data, yfile, default_flow_style=False)


if workflow_type == 'local':
    cmd = ' '.join([
        'snakemake --verbose --nolock',
        '--snakefile %s ' % snakefile,
        '--local-cores %s ' % local_cores,
        '--jobs %s ' % jobs,
        '--configfile %s ' % config_file,
        '--latency-wait %d' % latency_wait
    ])

elif workflow_type == 'torque':
    eo = '-e {cluster.error} -o {cluster.output} '

    cluster_setup = '\" qsub %s\
                     -l nodes=1:ppn={cluster.n} \
                     -l mem={cluster.mem}gb \
                     -l walltime={cluster.time}\" ' % eo

    cmd = ' '.join(['snakemake --verbose --nolock',
                    '--snakefile %s ' % snakefile,
                    '--local-cores %s ' % local_cores,
                    '--cores %s ' % cores,
                    '--jobs %s ' % jobs,
                    '--restart-times %d' % restart_times,
                    '--keep-going',
                    '--cluster-config %s ' % cluster_config,
                    '--cluster %s '  % cluster_setup,
                    '--configfile %s ' % config_file,
                    '--latency-wait %d' % latency_wait
                ])

elif workflow_type == "profile":
    cmd = ' '.join(['snakemake --nolock',
                    '--snakefile %s ' % snakefile,
                    '--cluster-config %s ' % cluster_config,
                    '--profile %s '  % profile,
                    '--configfile %s ' % config_file
                    ]
                   )

else:
    ValueError('Incorrect workflow specified:', workflow_type)

print(cmd)
proc = subprocess.Popen(cmd, shell=True)
proc.wait()
