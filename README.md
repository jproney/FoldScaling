# Test-time scaling for protein diffusion models

## How to Run Boltz Experiments Zero Order vs. Random Search on Monomers Data

Install boltz by running:

```bash
cd /path/to/FoldScaling/boltz
```

```bash
pip install -e .
```

First, go to the `boltz_monomers` directory. This is required so that if you are running an experiment that needs msa, the script can find the `.a3m` file.

```bash
cd /path/to/FoldScaling
```

```bash
cd data/boltz_monomers
```

To download the cif files for all the monomers, run:
```bash
boltz-utils dld-cif ../monomers.txt ../ground_truth_cif/
```
Where the first argument is a txt file with all the PDB IDs, and the second argument is the desired output directory to download all the cif files.

## Experiments Ran

You can copy these into multiple `.sh` files and put them inside `data/boltz_monomers/`:

```bash
#!/bin/bash
set -e

systemd-inhibit bash -c "
  boltz-exp monomers-predict --data_dir . --num_random_samples 2 --num_neighbors 2 --num_iterations 1 --verifier plddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 8 --num_neighbors 2 --num_iterations 4 --verifier plddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 16 --num_neighbors 2 --num_iterations 8 --verifier plddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 32 --num_neighbors 2 --num_iterations 16 --verifier plddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 64 --num_neighbors 2 --num_iterations 32 --verifier plddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 128 --num_neighbors 2 --num_iterations 64 --verifier plddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 256 --num_neighbors 2 --num_iterations 128 --verifier plddt --num_monomers 25 &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 512 --num_neighbors 2 --num_iterations 256 --verifier plddt --num_monomers 25
"
```

```bash
#!/bin/bash
set -e

systemd-inhibit bash -c "
  boltz-exp monomers-predict --data_dir . --num_random_samples 2 --num_neighbors 2 --num_iterations 1 --verifier lddt --num_monomers 25 --gt_cifs ../ground_truth_cif/ &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 8 --num_neighbors 2 --num_iterations 4 --verifier lddt --num_monomers 25 --gt_cifs ../ground_truth_cif/ &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 16 --num_neighbors 2 --num_iterations 8 --verifier lddt --num_monomers 25 --gt_cifs ../ground_truth_cif/ &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 32 --num_neighbors 2 --num_iterations 16 --verifier lddt --num_monomers 25 --gt_cifs ../ground_truth_cif/ &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 64 --num_neighbors 2 --num_iterations 32 --verifier lddt --num_monomers 25 --gt_cifs ../ground_truth_cif/ &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 128 --num_neighbors 2 --num_iterations 64 --verifier lddt --num_monomers 25 --gt_cifs ../ground_truth_cif/ &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 256 --num_neighbors 2 --num_iterations 128 --verifier lddt --num_monomers 25 --gt_cifs ../ground_truth_cif/ &&
  boltz-exp monomers-predict --data_dir . --num_random_samples 512 --num_neighbors 2 --num_iterations 256 --verifier lddt --num_monomers 25 --gt_cifs ../ground_truth_cif/
"
```

```bash
#!/bin/bash
set -e

systemd-inhibit bash -c '
  boltz-exp monomers-single-sample --data_dir . --denoising_steps 200 --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . --denoising_steps 800 --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . --denoising_steps 1600 --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . --denoising_steps 3200 --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . --denoising_steps 6400 --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . --denoising_steps 12800 --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . --denoising_steps 25600 --num_monomers 25 &&
  boltz-exp monomers-single-sample --data_dir . --denoising_steps 51200 --num_monomers 25
'
```

## How to Generate Plots

Assuming you are directly in `FoldScaling`, to generate results using plddt verifier you can run:
```bash
boltz-results plot-nfe-vs-plddt --results results/
```

To generate a close-up of just the zero order and random sampling results, you can run:
```bash
boltz-results plot-nfe-vs-plddt --results results/ --no_show_single
```

Assuming you are directly in `FoldScaling`, to generate results using lddt oracle verifier you can run:
```bash
boltz-results plot-nfe-vs-lddt --results results/ --gt data/ground_truth_cif/
```
This will print the table, and show the line plot of nfe vs lddt as well as the distributions of the scores for each method: zero order, random sampling, and varying the denoising steps.

To generate a close-up of just the zero order and random sampling results, you can run:
```bash
boltz-results plot-nfe-vs-lddt --results results/ --gt data/ground_truth_cif/ --no_show_single
```