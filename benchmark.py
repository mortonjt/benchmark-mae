import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from biom.util import biom_open
from benchmark_mae.generators import deposit, random_multimodal
import yaml


# snakemake config
config_file = 'params.yaml'
workflow_type = 'local'
local_cores = 1
cores = 4
jobs = 1
force = True
snakefile = 'Snakefile'
dry_run = False
output_dir = 'effect_size_small/'
quiet=False
keep_logger=True
cluster_config = 'cluster.json'
latency_wait = 120
profile = os.path.dirname(__file__)
restart_times = 1

# simulation parameters
regenerate_simulations = False
num_samples = 50
num_features = 300
#mu_num = 8            # mean of the numerator taxa
mu_null = 0            # mean of the common taxa
#mu_denom = 2          # mean of the denominator taxa
max_diff = 4           # largest separation between the normals
min_diff = 0.5         # smallest separation between the normals
min_alpha = 3          # smallest sequencing depth
max_alpha = 9          # largest sequencing depth
min_bias = 0.1         # smallest feature bias variance
max_bias = 3           # largest feature bias variance
min_null = 0.9         # smallest proportion of null species
max_null = 0.1         # largest proportion of null species
min_ratio = 1          # smallest differential species ratio
max_ratio = 5          # largest differential species ratio
sigma = 0.5            # variance of the random effects distribution
pi1 = 0.1              # percentage of the species
pi2 = 0.3              # percentage of the species
low = 0                # lower value for spectrum
high = 5               # higher value for the spectrum
spread = 2             # variance of unimodal species distribution
feature_bias = 1       # species bias
alpha = 6              # global sampling depth
seed = None            # random seed

# benchmark parameters
top_N = 50     # top hits to evaluate
intervals = 3
benchmark = 'effect_size'
oreps = 2
tools = ['deep_mae', 'pearson', 'spearman']

sample_ids = []
if regenerate_simulations:
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    for i, ef in enumerate(np.linspace(low, high, intervals)):
        for r in range(reps):
            sample_id = i
            ef = np.round(ef, decimals=2)
            print('ef', ef, 'r', r)
            res = random_multimodal(
                num_microbes=100, num_metabolites=1000, num_samples=200,
                num_latent_microbes=3, num_latent_metabolites=3,
                num_latent_shared=6, low=-1, high=1,
                microbe_total=1000, metabolite_total=10000,
                uB=ef, sigmaB=1, sigmaQ=0.1,
                uU1=0, sigmaU1=1, uU2=0, sigmaU2=1,
                uV1=0, sigmaV1=1, uV2=0, sigmaV2=1,
                seed=seed)
            microbe_counts, metabolite_counts, X, Q, U1, U2, V1, V2 = res
            # setup metadata
            X = pd.DataFrame(X, index=microbe_counts.index,
                             columns=['X%d' % d for d in range(X.shape[1])])
            Q = pd.DataFrame(Q, index=microbe_counts.index,
                             columns=['Q%d' % d for d in range(Q.shape[1])])
            metadata = pd.concat((X, Q), axis=1)
            metadata['effect_size'] = ef
            deposit(microbe_counts, metabolite_counts,
                    metadata, U1, U2, V1, V2,
                    sample_id, r, output_dir)
            sample_ids.append(sample_id)

    # generate config file
    data = {'benchmark': benchmark,
            'intervals': intervals,
            'output_dir': output_dir,
            'samples': sample_ids,
            'reps': reps,
            'tools': tools,
            'top_N': top_N}
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
