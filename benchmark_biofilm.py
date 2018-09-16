import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from biom.util import biom_open
from benchmark_mae.generators import (
    deposit, deposit_biofilms, random_biofilm
)
from benchmark_mae.sim import cystic_fibrosis_simulation

import yaml


# snakemake config
iteration = 3
config_file = 'test_cf_params%d.yaml' % iteration
workflow_type = 'local'
local_cores = 1
cores = 4
jobs = 1
force = True
snakefile = 'Snakebiofilm'
dry_run = False
output_dir = 'test_cf_benchmark%d/' % iteration
quiet=False
keep_logger=True
cluster_config = 'cluster.json'
latency_wait = 120
profile = os.path.dirname(__file__)
restart_times = 1

# simulation parameters
regenerate_simulations = True

num_metabolites = 2
num_microbes = 2
num_samples = 126

uU = 0
sigmaU = 1
uV = 0
sigmaV = 1
latent_dim = 3
sigmaQmin = 1
sigmaQmax = 3

microbe_total = 5e2
metabolite_total = 10e8

microbe_tau = 0.1
metabolite_tau = 0.1

microbe_kappa = 0.1
metabolite_kappa = 0.1

min_time = 0
max_time = 10
min_y = 18
max_y = 20
seed = None

# benchmark parameters
top_OTU = 20      # top OTUs to evaluate
top_MS = 20       # top metabolites to evaluate

intervals = 1
benchmark = 'effect_size'
reps = 1
tools = ['deep_mae', 'pearson', 'spearman']

sigmaQ = np.linspace(sigmaQmin, sigmaQmax, intervals)
sigmaQ = list(map(float, sigmaQ.tolist()))
sample_ids = []
choice = 'abcdefghijklmnopqrstuvwxyz'


# retrieve absolute abundances
df = cystic_fibrosis_simulation('benchmark_mae/data')
table = df.loc[
    np.logical_and(
        np.logical_and(
            df.y > min_y, df.y < max_y
        ),
        np.logical_and(
            df.time > min_time, df.time < max_time
        )
    )
]

if regenerate_simulations:

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)

    for i, s in enumerate(sigmaQ):
        for r in range(reps):
            sample_id = '%d_%s' % (s, choice[r])

            res = random_biofilm(
                table, uU=uU, sigmaU=sigmaU, uV=uV, sigmaV=sigmaV,
                sigmaQ=s, latent_dim=latent_dim,
                num_microbes=num_microbes,
                num_metabolites=num_metabolites,
                microbe_total=microbe_total,
                microbe_kappa=microbe_kappa,
                microbe_tau=microbe_tau,
                metabolite_total=metabolite_total,
                metabolite_kappa=metabolite_kappa,
                metabolite_tau=metabolite_tau,
                seed=seed)
            edges, microbe_counts, metabolite_counts = res

            deposit_biofilms(output_dir=output_dir,
                             table1=microbe_counts,
                             table2=metabolite_counts,
                             edges=edges,
                             metadata=table,
                             sample_id=sample_id
            )

            sample_ids.append(sample_id)

    # generate config file
    data = {'benchmark': benchmark,
            'intervals': intervals,
            'output_dir': output_dir,
            'samples': list(sample_ids),
            'reps': reps,
            'tools': tools,
            'top_MS': top_MS,
            'top_OTU': top_OTU,
            'min_y': min_y,
            'max_y': max_y,
            'min_time': min_time,
            'max_time': max_time,
            # parameters to simulate the model
            'num_samples' : num_samples,
            'num_microbes' : num_microbes,
            'num_metabolites' : num_metabolites,
            'microbe_total' : microbe_total,
            'metabolite_total' : metabolite_total,
            'latent_dim' : latent_dim,
            'sigmaQ' : list(sigmaQ),
            'uU' : uU,
            'sigmaU' : sigmaU,
            'uV' : uV,
            'sigmaV' : sigmaV
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
