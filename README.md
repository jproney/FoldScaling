# Test-time scaling for protein diffusion models

## How to Run Boltz Experiments Zero Order vs. Random Search on Monomers Data

First, go to the boltz_monomers directory. This is required so that if you are running an experiment that needs msa, the script can find the a3m file.

```bash
cd /path/to/FoldScaling
```

```bash
cd data/boltz_monomers
```

Next, run this script:
```bash
boltz monomers-predict --data_dir . --use_msa --sampling steps 100 --recycling_steps 3 --num_random_samples 64 --num_neighbors 8 --num_iterations 8
```

This is an example of an experiment. Providing data directory is required, and in this case it is just the current working directory, `.`. In this experiment, we will use msa, run 100 denoising steps, 3 recycling steps, 64 random samples for random sampling, use 8 neighbors and 8 iterations for zero order search.

The results will be saved inside `FoldScaling/results/boltz_monomers_msa_False_sampling_100_recycling_3_random_samples_64_neighbors_8_iterations_8`.

To generate a plot of the results, run:
```bash
boltz plot-plddt-diffs <path/to/boltz_results>
```
Where the path is the path to the results you generated in the previous command.