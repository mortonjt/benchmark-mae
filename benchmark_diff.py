import os
import shutil
import subprocess
import numpy as np
import pandas as pd
from biom.util import biom_open
from benchmark_mae.generators import (
    deposit, deposit_biofilms, deposit_blocktable, random_biofilm, random_block_table
)
from benchmark_mae.sim import cystic_fibrosis_simulation

import yaml


iteration = 2
workflow_type = 'local'
local_cores = 1
cores = 4
jobs = 1
force = True
dry_run = False
output_dir = 'test_diff_abundance_benchmark%d/' % iteration
config_file = output_dir + 'test_diff_abundance_params%d.yaml' % iteration
quiet=False
keep_logger=True
cluster_config = 'cluster.json'
latency_wait = 120
profile = os.path.dirname(__file__)
restart_times = 1

# simulation parameters
regenerate_simulations = True


uU = 0
sigmaU = 1
uV = 0
sigmaV = 1
latent_dim = 3
sigmaQmin = 1
sigmaQmax = 3

microbe_total = 1e5
library_size = 1000
microbe_tau = 1.1
microbe_kappa = 1.4

min_time = 0
max_time = 9
min_y = 15
max_y = 20
seed = None

# benchmark parameters
top_OTU = 20      # top OTUs to evaluate
top_MS = 20       # top metabolites to evaluate

intervals = 1
benchmark = 'effect_size'
reps = 1
tools = ['ttest', 'multinomial']
modes = ['abs', 'rel']

sample_ids = []
choice = 'abcdefghijklmnopqrstuvwxyz'
library_size = 10000
effect_sizes = [0.5, 1]
species_vars = [0.5, 3]
microbe_kappa = 0.5
microbe_tau = 0.5
class_size = 100  # 200 samples
num_samples = class_size * 2
num_microbes = 300

if regenerate_simulations:

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    for s, sv in enumerate(species_vars):
        for i, ef in enumerate(effect_sizes):

            sample_id = '%d_%d' % (s, i)
            res = random_block_table(class_size, num_microbes,
                                     species_var=sv,
                                     microbe_kappa=microbe_kappa,
                                     microbe_tau=microbe_tau,
                                     library_size=library_size,
                                     microbe_total=microbe_total,
                                     effect_size=ef)

            abs_table, rel_table, metadata, ground_truth = res
            deposit_blocktable(output_dir, abs_table, rel_table,
                               metadata, ground_truth, sample_id)
            sample_ids.append(sample_id)

    # generate config file
    data = {
        'benchmark': benchmark,
        'intervals': intervals,
        'output_dir': output_dir,
        'samples': list(sample_ids),
         # 'reps': reps,
        'tools': tools,
        'modes': modes,
         # parameters to simulate the model
        'species_vars': species_vars,
        'effect_sizes': effect_sizes,
        'num_samples' : num_samples,
        'num_microbes' : num_microbes,
        'microbe_total' : microbe_total,
    }
    with open(config_file, 'w') as yfile:
        yaml.dump(data, yfile, default_flow_style=False)


if workflow_type == 'local':
    from jobarray import local_cmd
    cmd = ';'.join(local_cmd(output_dir, sample_ids, tools, modes,
                             'sampleids', concurrent_jobs=jobs))

elif workflow_type == 'jobarray':
    from jobarray import jobarray_cmd
    cmd = ' '.join(jobarray_cmd(output_dir, sample_ids, tools, modes,
                                'sampleids', concurrent_jobs=jobs))


else:
    ValueError('Incorrect workflow specified:', workflow_type)
print(cmd)
proc = subprocess.Popen(cmd, shell=True)
proc.wait()
