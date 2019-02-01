from itertools import product

def local_cmd(directory, sample_ids, tools, modes, fname, concurrent_jobs=10):
    opts = list(product(sample_ids, tools, modes))

    # careful about overwriting here.
    text = '\n'.join(
        list(
            map(lambda x: "%s\t%s\t%s" % x, opts
            )
        )
    )

    with open(directory + fname, 'w') as fh:
        fh.write(text)

    cmd = [
        "sample=$(head -n ${PBS_ARRAYID} %s | tail -n 1 | cut -f 1);" % fname,
        "tool=$(head -n ${PBS_ARRAYID}  %s | tail -n 1 | cut -f 2);" % fname,
        "mode=$(head -n ${PBS_ARRAYID}  %s | tail -n 1 | cut -f 3);" % fname ,
        ("run_models.py run_${tool} --table-file %{mode}_table.${sample}.biom "
         "--metadata-file metadata.${sample}.txt --output-file ${tool}.${sample}.results\'"),
    ]
    return cmd


def jobarray_cmd(directory, sample_ids, tools, modes, fname, concurrent_jobs=10):
    opts = list(product(sample_ids, tools, modes))

    # careful about overwriting here.
    text = '\n'.join(
        list(
            map(lambda x: "%s\t%s\t%s" % x, opts
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
        "mode=$(head -n ${PBS_ARRAYID}  %s | tail -n 1 | cut -f 3);" % fname ,
        ("echo $tool; run_models.py run_${tool} --table-file %{mode}_table.${sample}.biom "
         "--metadata-file metadata.${sample}.txt --output-file ${tool}.${sample}.results\'"),
        "| qsub -N benchmark",
        "-l nodes=1:ppn=4",
        "-l cput=72:30:00",
        "-m abe",
        "-M jmorton@eng.ucsd.edu",
        "-t 1-%d" % (len(opts)) + "%" + str(concurrent_jobs)
    ]
    return cmd
