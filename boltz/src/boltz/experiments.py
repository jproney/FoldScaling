import gc
import pathlib

import click
import torch
from tqdm import tqdm

import shutil
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from boltz.search_algorithms import (
    random_sampling,
    zero_order_sampling,
    search_over_paths,
)

warnings.simplefilter("ignore", PDBConstructionWarning)


@click.group()
def cli() -> None:
    """Experiments with Boltz-1."""
    return


@cli.command()
@click.option(
    "--data_dir",
    type=click.Path(exists=True),
    help="The path to the directory containing the fasta files.",
)
@click.option(
    "--use_msa",
    is_flag=True,
    help="Whether to use the MSA file.",
    default=False,
)
@click.option(
    "--denoising_steps",
    type=int,
    help="The number of denoising steps to use for prediction.",
    default=100,
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction.",
    default=0,
)
@click.option(
    "--num_random_samples",
    type=int,
    help="The number of random samples to use for prediction.",
    default=64,
)
@click.option(
    "--num_neighbors",
    type=int,
    help="The number of neighbors to use for zero-order search.",
    default=8,
)
@click.option(
    "--num_iterations",
    type=int,
    help="The number of iterations to use for zero-order search.",
    default=8,
)
@click.option(
    "--num_monomers",
    type=int,
    help="The number of monomers to use for prediction.",
    default=50,
)
@click.option(
    "--verifier",
    type=str,
    help="The score function to use for prediction.",
    default="plddt",
)
@click.option(
    "--gt_cifs",
    type=str,
    help="The path to the directory containing the ground truth cif files.",
    default=None,
)
def monomers_predict(
    data_dir: str,
    use_msa: bool,
    denoising_steps: int,
    recycling_steps: int,
    num_random_samples: int,
    num_neighbors: int,
    num_iterations: int,
    num_monomers: int,
    verifier: str,
    gt_cifs: str,
) -> None:
    """Make sure to run this command inside the data directory."""

    parent_dir = pathlib.Path(data_dir).absolute().parent.parent
    results_dir = parent_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_dir = (
        results_dir
        / f"boltz_monomers_msa_{use_msa}_denoising_{denoising_steps}_recycling_{recycling_steps}_random_samples_{num_random_samples}_neighbors_{num_neighbors}_iterations_{num_iterations}"
    )
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fasta_files = list(pathlib.Path(data_dir).glob("*.fasta"))

    if use_msa:
        fasta_files = [f for f in all_fasta_files if "_no_msa" not in f.name]
    else:
        fasta_files = [f for f in all_fasta_files if "_no_msa" in f.name]

    fasta_files = sorted(fasta_files)[:num_monomers]
    fasta_files_unprocessed = []

    for fasta in fasta_files:
        fasta_name = fasta.stem
        sub_dir = out_dir / fasta_name

        if sub_dir.exists():
            random_dir = sub_dir / f"random_{fasta_name}"
            zero_order_dir = sub_dir / f"zero_order_{fasta_name}"

            random_exists = False
            if random_dir.exists():
                pred_dir = random_dir / "predictions" / fasta_name
                print(pred_dir)
                if pred_dir.exists():
                    random_exists = True

            zero_order_exists = False
            if zero_order_dir.exists():
                pred_dir = zero_order_dir / "predictions" / fasta_name
                if pred_dir.exists():
                    zero_order_exists = True

            if not random_exists or not zero_order_exists:
                fasta_files_unprocessed.append(fasta)
        else:
            fasta_files_unprocessed.append(fasta)

    print(f"Unprocessed FASTA files: {len(fasta_files_unprocessed)}")

    for fasta in tqdm(fasta_files_unprocessed, desc="Processing monomers"):
        print(f"\n------\nProcessing {fasta.name}")

        fasta_name = fasta.stem
        sub_dir = out_dir / fasta_name
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)

        random_sampling(
            data=str(fasta),
            out_dir=str(sub_dir),
            devices=1,
            accelerator="gpu",
            sampling_steps=denoising_steps,
            step_scale=1.638,
            write_full_pae=True,
            write_full_pde=False,
            output_format="mmcif",
            num_workers=2,
            override=False,
            seed=None,
            use_msa_server=False,
            no_potentials=True,
            recycling_steps=recycling_steps,
            num_random_samples=num_random_samples,
            verifier=verifier,
            gt_cifs=gt_cifs,
        )

        zero_order_sampling(
            data=str(fasta),
            out_dir=str(sub_dir),
            devices=1,
            accelerator="gpu",
            sampling_steps=denoising_steps,
            step_scale=1.638,
            write_full_pae=True,
            write_full_pde=False,
            output_format="mmcif",
            num_workers=2,
            override=False,
            seed=None,
            use_msa_server=False,
            no_potentials=True,
            recycling_steps=recycling_steps,
            num_candidates=num_neighbors,
            num_iterations=num_iterations,
            verifier=verifier,
            gt_cifs=gt_cifs,
        )

        # Free memory after each FASTA file
        torch.cuda.empty_cache()
        gc.collect()


@cli.command()
@click.option(
    "--data_dir",
    type=click.Path(exists=True),
    help="The path to the directory containing the fasta files.",
)
@click.option(
    "--use_msa",
    is_flag=True,
    help="Whether to use the MSA file.",
    default=False,
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction.",
    default=0,
)
@click.option(
    "--num_monomers",
    type=int,
    help="The number of monomers to use for prediction.",
    default=50,
)
@click.option(
    "--denoising_steps",
    type=int,
    help="The number of denoising steps to use for prediction.",
    default=200,
)
def monomers_single_sample(
    data_dir: str,
    use_msa: bool,
    recycling_steps: int,
    num_monomers: int,
    denoising_steps: int,
) -> None:
    """Make sure to run this command inside the data directory."""

    parent_dir = pathlib.Path(data_dir).absolute().parent.parent
    results_dir = parent_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    all_fasta_files = list(pathlib.Path(data_dir).glob("*.fasta"))

    if use_msa:
        fasta_files = [f for f in all_fasta_files if "_no_msa" not in f.name]
    else:
        fasta_files = [f for f in all_fasta_files if "_no_msa" in f.name]

    fasta_files = sorted(fasta_files)[:num_monomers]

    out_dir = (
        results_dir
        / f"boltz_monomers_msa_{use_msa}_denoising_{denoising_steps}_recycling_{recycling_steps}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for fasta in tqdm(fasta_files, desc=f"Sampling={denoising_steps}", leave=False):
        print(f"\n------\nProcessing {fasta.name}")

        random_sampling(
            data=str(fasta),
            out_dir=str(out_dir),
            devices=1,
            accelerator="gpu",
            sampling_steps=denoising_steps,
            step_scale=1.638,
            write_full_pae=True,
            write_full_pde=False,
            output_format="mmcif",
            num_workers=2,
            override=False,
            seed=None,
            use_msa_server=False,
            no_potentials=True,
            recycling_steps=recycling_steps,
            num_random_samples=1,
            verifier='plddt',
        )

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    cli()
