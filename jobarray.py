from itertools import product


def jobarray_cmd(directory, sample_ids, tools, fname, concurrent_jobs=10):
    opts = list(product(sample_ids, tools))

    # careful about overwriting here.
    text = '\n'.join(
        list(
            map(lambda x: "%s\t%s" % x, opts
            )
        )
    )

    with open(directory + fname, 'w') as fh:
        fh.write(text)            


    cmd = [
        "echo \'module load python_3.5.5;",
        "cd /home/mortonjt/Documents/benchmark-mae/%s;" % directory,
        "sample=$(head -n ${PBS_ARRAYID} %s | tail -n 1 | cut -f 1);" % fname,
        "tool=$(head -n ${PBS_ARRAYID}  %s | tail -n 1 | cut -f 2);" % fname,
        ("echo $tool; run_models.py run_${tool} --table1-file table_microbes.${sample}.biom "
         "--table2-file table_metabolites.${sample}.biom --output-file ${tool}.${sample}.results\'"),
        "| qsub -N benchmark",
        "-l nodes=1:ppn=4",
        "-l cput=72:30:00",
        "-m abe",
        "-M jmorton@eng.ucsd.edu",
        "-t 1-%d" % (len(opts)) + "%" + str(concurrent_jobs)
    ]
    return cmd
