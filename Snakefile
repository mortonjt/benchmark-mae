import os
import tempfile
import numpy as np
from itertools import product


benchmark = config['benchmark']
output_dir = config['output_dir']
intervals = config['intervals']
top_OTU = config['top_OTU']
top_MS = config['top_MS']
reps = config['reps']
TOOLS = config['tools']

choice = 'abcdefghijklmnopqrstuvwxyz'
REPLICATES = list(choice[:reps])
SAMPLES = np.arange(intervals).astype(np.str)
SAMPLES = list(map(lambda x: '%s_%s' % x, product(SAMPLES, REPLICATES)))


rule all:
    input:
        output_dir + "confusion_matrix.summary"


rule run_tool:
    input:
        microbes = output_dir + "table_microbes.{sample}.biom",
        metabolites = output_dir + "table_metabolites.{sample}.biom",
        metadata = output_dir + "metadata.{sample}.txt"
    output:
        output_dir + "{tool}.{sample}.results"
    run:
        shell("""
        run_models.py run_{wildcards.tool} \
            --table1-file {input.microbes} \
            --table2-file {input.metabolites} \
            --output-file {output}
        """)

rule summarize:
    input:
        results = expand(output_dir + "{tool}.{sample}.results",
                         tool=TOOLS, sample=SAMPLES),
        truths = expand(output_dir + "ranks.{sample}.txt", sample=SAMPLES),
        params = expand(output_dir + "B.{sample}.txt", sample=SAMPLES)
    output:
        output_dir + "{tool}.summary"
    run:
        from benchmark_mae.evaluate import top_absolute_results
        top_absolute_results(input.results, input.truths, input.params,
                             output[0], top_OTU, top_MS)


rule aggregate_summaries:
    input:
        summaries = expand(output_dir + "{tool}.summary", tool=TOOLS),
        metadata = expand(output_dir + "metadata.{sample}.txt", sample=SAMPLES),
    output:
        output_dir + "confusion_matrix.summary"
    run:
        from benchmark_mae.evaluate import aggregate_summaries
        aggregate_summaries(input.summaries, input.metadata,
                            benchmark, output[0])


