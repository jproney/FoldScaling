# Test-time scaling for protein diffusion models

## How to Run Boltz Experiments Zero Order vs. Random Search on Monomers Data

First, go to the boltz_monomers directory. This is required so that if you are running an experiment that needs msa, the script can find the a3m file.

```bash
cd /path/to/FoldScaling
```

```bash
cd data/boltz_monomers
```

To run the **zero order sampling vs random sampling experiment** for all monomers, run:
```bash
boltz-exp monomers-predict --data_dir . --use_msa --denoising_steps 100 --recycling_steps 3 --num_random_samples 64 --num_neighbors 8 --num_iterations 8
```

This is an example of an experiment. Providing data directory is required, and in this case it is just the current working directory, `.`. In this experiment, we will use msa, run 100 denoising steps, 3 recycling steps, 64 random samples for random sampling, use 8 neighbors and 8 iterations for zero order search.

The results will be saved inside `FoldScaling/results/boltz_monomers_msa_True_denoising_100_recycling_3_random_samples_64_neighbors_8_iterations_8`.

To generate a plot of the results, run:
```bash
boltz-exp plot-results <path/to/boltz_results>
```
Where the path is the path to the results you generated in the previous command.

To download the cif files for all the monomers, run:
```bash
boltz-utils dld-cif path/to/data/monomers.txt path/to/data/ground_truth_cif/
```
Where the first argument is a txt file with all the PDB IDs, and the second argument is the desired output directory to download all the cif files.

To generate a table of the results from the above experiment, run (as an example):
```bash
boltz-exp avg-monomers-search path/to/results path/to/data/ground_truth_cif
```
This will print a table of the average pLDDT, average pTM, average confidence, and average LDDT scores across the desired monomers for random sampling and zero order sampling, respectively.

To run the **single sample experiment**, run:
```bash
cd data/boltz_monomers
```

And then (as an example):
```bash
boltz-exp monomers-single-sample --data_dir . --use_msa --denoising_steps 1600 --recycling_steps 3
```
The results will be saved inside `FoldScaling/results/boltz_monomers_msa_True_denoising_1600_recycling_3`.

To generate a table of the results from the above experiment, run:
```bash
boltz-exp avg-monomers-single path/to/results path/to/data/ground_truth_cif
```
Assuming the path given only contains the results from varying the number of denoising steps.


## Experiments Ran

You can copy these into multiple `.sh` files and put them inside `data/boltz_monomers/`:
```bash
#!/bin/bash
set -e  # Exit on any error

systemd-inhibit boltz-exp monomers-predict \
  --data_dir . \
  --num_random_samples 256 \
  --num_neighbors 2 \
  --num_iterations 128
```

```bash
#!/bin/bash
set -e  # Exit on any error

systemd-inhibit bash -c "
  boltz-exp monomers-predict --data_dir . --num_random_samples 2 --num_neighbors 2 --num_iterations 1 --verifier pddlt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 8 --num_neighbors 2 --num_iterations 4 --verifier pddlt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 16 --num_neighbors 2 --num_iterations 8 --verifier pddlt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 32 --num_neighbors 2 --num_iterations 16 --verifier pddlt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 64 --num_neighbors 2 --num_iterations 32 --verifier pddlt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 128 --num_neighbors 2 --num_iterations 64 --verifier pddlt --num_monomers 25
"
```

```bash
#!/bin/bash
set -e  # Exit on any error

systemd-inhibit bash -c "
  boltz-exp monomers-predict --data_dir . --num_random_samples 2 --num_neighbors 2 --num_iterations 1 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 8 --num_neighbors 2 --num_iterations 4 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 16 --num_neighbors 2 --num_iterations 8 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 32 --num_neighbors 2 --num_iterations 16 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 64 --num_neighbors 2 --num_iterations 32 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 128 --num_neighbors 2 --num_iterations 64 --verifier lddt --num_monomers 25
"
```

```bash
#!/bin/bash
set -e  # Exit on any error

systemd-inhibit bash -c "
  boltz-exp monomers-single-sample --data_dir . -denoising_steps 200 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . -denoising_steps 800 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . -denoising_steps 1600 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . -denoising_steps 3200 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . -denoising_steps 6400 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . -denoising_steps 12800 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . -denoising_steps 25600 --verifier lddt --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . -denoising_steps 51200 --verifier lddt --num_monomers 25 &&
"
```