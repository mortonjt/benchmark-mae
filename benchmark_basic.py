import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from biom.util import biom_open
from benchmark_mae.generators import (
    deposit, random_multimodal, random_sigmoid_multimodal
)
import yaml


# snakemake config
iteration = 1
config_file = 'basic_params%d.yaml' % iteration
workflow_type = 'local'
local_cores = 1
cores = 4
jobs = 1
force = True
snakefile = 'Snakefile'
dry_run = False
output_dir = 'test_basic_benchmark%d/' % iteration
quiet=False
keep_logger=True
cluster_config = 'cluster.json'
latency_wait = 120
profile = os.path.dirname(__file__)
restart_times = 1

# simulation parameters
regenerate_simulations = True
num_samples = 100
num_microbes = 20
num_metabolites = 50
microbe_total = 5000
metabolite_total = 100000

# note deep mae stands out in small effect size areas
latent_dim = 2
uB=0; sigmaB = 1; sigmaQ = 0.1
uU=0; sigmaU = 1; uV = 0; sigmaV = 2
low=-1; high=1
seed = None            # random seed

ef_low = 0.1           # lower value for spectrum
ef_high = 5            # higher value for the spectrum

# benchmark parameters
top_OTU = 50      # top OTUs to evaluate
top_MS = 20       # top metabolites to evaluate

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

    for i, ef in enumerate(np.linspace(ef_low, ef_high, intervals)):
        for r in range(reps):
            sample_id = i
            ef = np.round(ef, decimals=2)
            print(ef)
            res = random_multimodal(
                uB=uB, sigmaB = ef, sigmaQ = sigmaQ,
                uU=uU, sigmaU = sigmaU, uV = uV, sigmaV = sigmaV,
                num_microbes=num_microbes, num_metabolites=num_metabolites,
                num_samples=num_samples,
                latent_dim=latent_dim, low=low, high=high,
                microbe_total=microbe_total, metabolite_total=metabolite_total,
                seed=seed)

            microbe_counts, metabolite_counts, X, B, U, V = res
            # setup metadata
            X = pd.DataFrame(X, index=microbe_counts.index,
                             columns=['X%d' % d for d in range(X.shape[1])])
            metadata = X
            metadata['effect_size'] = ef
            deposit(output_dir, microbe_counts, metabolite_counts,
                    metadata, U, V, B,
                    sample_id, int(r))
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
            'uB' : uB,
            'sigmaB' : sigmaB,
            'sigmaQ' : sigmaQ,
            'uU' : uU,
            'sigmaU' : sigmaU,
            'uV' : uV,
            'sigmaV' : sigmaV,
            'low' : low,
            'high' : high,
            'ef_low' : ef_low,
            'ef_high' : ef_high
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
