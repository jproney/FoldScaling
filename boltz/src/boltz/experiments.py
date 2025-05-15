import gc
import json
import pathlib

import click
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import torch
import numpy as np
from tqdm import tqdm

import shutil
from Bio.PDB.MMCIFParser import MMCIFParser
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

from boltz.search_algorithms import (
    random_sampling,
    zero_order_sampling,
    search_over_paths,
    plddt_score,
)

from boltz.utils import (
    compute_lddt
)

warnings.simplefilter('ignore', PDBConstructionWarning)

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
def monomers_predict(
    data_dir: str,
    use_msa: bool,
    denoising_steps: int,
    recycling_steps: int,
    num_random_samples: int,
    num_neighbors: int,
    num_iterations: int,
    num_monomers: int,
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
    all_fasta_files = sorted(all_fasta_files)[:num_monomers]

    if use_msa:
        fasta_files = [f for f in all_fasta_files if "_no_msa" not in f.name]
    else:
        fasta_files = [f for f in all_fasta_files if "_no_msa" in f.name]

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
            score_fn=plddt_score,
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
            score_fn=plddt_score,
        )

        # search_over_paths(
        #     data=str(fasta),
        #     out_dir=str(sub_dir),
        #     devices=1,
        #     accelerator="gpu",
        #     sampling_steps=denoising_steps,
        #     step_scale=1.638,
        #     output_format="mmcif",
        #     num_workers=2,
        #     seed=None,
        #     use_msa_server=False,
        #     no_potentials=True,
        #     recycling_steps=recycling_steps,
        #     diffusion_samples=1,
        #     num_initial_paths=4,
        #     path_width=5,
        #     search_start_sigma=10.0,
        #     backward_stepsize=0.5,
        #     forward_stepsize=0.5,
        #     diffusion_solver_steps=50,
        #     confidence_fk=False,
        #     device="cuda"
        # )

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
            score_fn=plddt_score,
        )

        torch.cuda.empty_cache()
        gc.collect()


@cli.command()
@click.argument("results_root", type=click.Path(exists=True))
def plot_results(results_root: str):
    root = pathlib.Path(results_root)
    save_path = root / "plots"
    save_path.mkdir(parents=True, exist_ok=True)

    differences = []
    random_plddts, zos_plddts = [], []
    random_ptms, zos_ptms = [], []

    subdirs = [d for d in root.iterdir() if d.is_dir() and d.name != "plots"]

    for subdir in tqdm(subdirs, desc="Loading scores for plots"):
        try:
            random_dir = next(subdir.glob("random_*"), None)
            zos_dir = next(subdir.glob("zero_order_*"), None)

            random_json = next(
                (random_dir / "predictions" / subdir.name).glob("*.json"), None
            )
            zos_json = next(
                (zos_dir / "predictions" / subdir.name).glob("*.json"), None
            )

            with open(random_json, "r") as f:
                random_data = json.load(f)
            with open(zos_json, "r") as f:
                zos_data = json.load(f)

            random_plddt = float(random_data["complex_plddt"])
            zos_plddt = float(zos_data["complex_plddt"])
            random_ptm = float(random_data["ptm"])
            zos_ptm = float(zos_data["ptm"])

            random_plddts.append(random_plddt)
            zos_plddts.append(zos_plddt)
            random_ptms.append(random_ptm)
            zos_ptms.append(zos_ptm)
            differences.append(zos_plddt - random_plddt)
        except Exception as e:
            print(f"Skipping {subdir.name} due to error: {e}")
            continue

    # Convert lists explicitly to numpy arrays
    random_plddts = np.array(random_plddts)
    zos_plddts = np.array(zos_plddts)
    random_ptms = np.array(random_ptms)
    zos_ptms = np.array(zos_ptms)
    differences = np.array(differences)

    # Helper function to plot histogram and KDE
    def plot_histogram_kde(data, color, title, xlabel, filename):
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=20, density=True, alpha=0.6, color=color, edgecolor="black")
        kde = gaussian_kde(data)
        x_range = np.linspace(min(data), max(data), 1000)
        plt.plot(x_range, kde(x_range), color="k", lw=2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(save_path / filename)
        plt.close()

    # Plot Difference histogram (no KDE necessary here)
    plt.figure(figsize=(8, 5))
    plt.hist(differences, bins=20, color="skyblue", edgecolor="black")
    plt.axvline(0, color="red", linestyle="--")
    plt.title("PLDDT Difference per Monomer (ZOS - Random)")
    plt.xlabel("Difference in Best PLDDT")
    plt.ylabel("Number of Monomers")
    plt.tight_layout()
    plt.savefig(save_path / "plddt_diff_histogram.png")
    plt.close()

    # Individual PLDDT distributions
    plot_histogram_kde(
        random_plddts,
        "skyblue",
        "Random Sampling: PLDDT Distribution",
        "PLDDT",
        "random_plddt_distribution.png",
    )
    plot_histogram_kde(
        zos_plddts,
        "lightgreen",
        "Zero-Order Search: PLDDT Distribution",
        "PLDDT",
        "zos_plddt_distribution.png",
    )

    # Individual PTM distributions
    plot_histogram_kde(
        random_ptms,
        "salmon",
        "Random Sampling: pTM Distribution",
        "pTM",
        "random_ptm_distribution.png",
    )
    plot_histogram_kde(
        zos_ptms,
        "orange",
        "Zero-Order Search: pTM Distribution",
        "pTM",
        "zos_ptm_distribution.png",
    )

    # Overlay PLDDT distributions
    plt.figure(figsize=(8, 5))
    plt.hist(
        random_plddts,
        bins=20,
        density=True,
        alpha=0.6,
        label="Random Sampling",
        color="skyblue",
        edgecolor="black",
    )
    plt.hist(
        zos_plddts,
        bins=20,
        density=True,
        alpha=0.6,
        label="Zero-Order Search",
        color="lightgreen",
        edgecolor="black",
    )
    x_range = np.linspace(
        min(np.min(random_plddts), np.min(zos_plddts)),
        max(np.max(random_plddts), np.max(zos_plddts)),
        1000,
    )
    plt.plot(x_range, gaussian_kde(random_plddts)(x_range), color="blue", lw=2)
    plt.plot(x_range, gaussian_kde(zos_plddts)(x_range), color="green", lw=2)
    plt.title("Overlay: PLDDT Distribution")
    plt.xlabel("PLDDT")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "overlay_plddt_distribution.png")
    plt.close()

    # Overlay PTM distributions
    plt.figure(figsize=(8, 5))
    plt.hist(
        random_ptms,
        bins=20,
        density=True,
        alpha=0.6,
        label="Random Sampling",
        color="salmon",
        edgecolor="black",
    )
    plt.hist(
        zos_ptms,
        bins=20,
        density=True,
        alpha=0.6,
        label="Zero-Order Search",
        color="orange",
        edgecolor="black",
    )
    x_range = np.linspace(
        min(np.min(random_ptms), np.min(zos_ptms)),
        max(np.max(random_ptms), np.max(zos_ptms)),
        1000,
    )
    plt.plot(x_range, gaussian_kde(random_ptms)(x_range), color="red", lw=2)
    plt.plot(x_range, gaussian_kde(zos_ptms)(x_range), color="darkorange", lw=2)
    plt.title("Overlay: pTM Distribution")
    plt.xlabel("pTM")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "overlay_ptm_distribution.png")
    plt.close()

    print(f"Saved all plots to: {save_path}")


@cli.command()
@click.option(
    "--results",
    type=click.Path(exists=True),
    help="The path to the directory containing the fasta files.",
)
@click.option(
    "--gt",
    type=click.Path(exists=True),
    help="The path to the directory containing the ground truth files.",
)
def avg_monomers_single(results: str, gt: str):
    """
    Compute average pLDDT, pTM, confidence score, and lDDT for each experiment.
    """

    def get_monomer_result(monomer_dir):
        pred_dir = (
            monomer_dir / "predictions" / "_".join(monomer_dir.name.split("_")[1:])
        )
        json_file = next(pred_dir.glob("*.json"))
        cif_file = next(pred_dir.glob("*.cif"))
        with open(json_file, "r") as f:
            data = json.load(f)
        return (
            float(data["complex_plddt"]),
            float(data["ptm"]),
            float(data["confidence_score"]),
            cif_file,
        )

    def summarize(denoising_steps, summary):
        (
            avg_plddt,
            avg_ptm,
            avg_conf,
            avg_lddt,
            std_plddt,
            std_ptm,
            std_conf,
            std_lddt,
        ) = summary[denoising_steps]
        print(f"\n{denoising_steps} Denoising Steps")
        print("-" * 30)
        print(f"Avg pLDDT: {avg_plddt:.4f} ± {std_plddt:.4f}")
        print(f"Avg pTM: {avg_ptm:.4f} ± {std_ptm:.4f}")
        print(f"Avg Confidence: {avg_conf:.4f} ± {std_conf:.4f}")
        print(f"Avg lDDT: {avg_lddt:.4f} ± {std_lddt:.4f}")

    root = pathlib.Path(results)
    sub_dirs = [
        d
        for d in root.iterdir()
        if d.is_dir() and d.name != "plots" and d.name != ".DS_Store"
    ]

    gt_files = sorted(
        [
            d
            for d in pathlib.Path(gt).iterdir()
            if not d.is_dir() and d.name != "plots" and d.name != ".DS_Store"
        ],
        key=lambda x: x.name,
    )
    gt_files = gt_files[:len(sub_dirs)]
    summary = {}

    for exp_dir in sub_dirs:
        name_parts = exp_dir.name.split("_")
        denoising_idx = name_parts.index("denoising")
        denoising_steps = int(name_parts[denoising_idx + 1])
        all_plddt, all_ptm, all_conf, all_lddt = [], [], [], []

        for subdir, gt_cif in zip(exp_dir.iterdir(), gt_files.iterdir()):
            plddt, ptm, conf, random_cif = get_monomer_result(subdir)
            all_plddt.append(plddt)
            all_ptm.append(ptm)
            all_conf.append(conf)
            all_lddt.append(compute_lddt(random_cif, gt_cif))

        if all_plddt:
            avg_plddt = np.mean(all_plddt)
            avg_ptm = np.mean(all_ptm)
            avg_conf = np.mean(all_conf)
            avg_lddt = np.mean(all_lddt)
            std_plddt = np.std(all_plddt)
            std_ptm = np.std(all_ptm)
            std_conf = np.std(all_conf)
            std_lddt = np.std(all_lddt)
            summary[denoising_steps] = (
                avg_plddt,
                avg_ptm,
                avg_conf,
                avg_lddt,
                std_plddt,
                std_ptm,
                std_conf,
                std_lddt,
            )

        print("\n" + "=" * 50)
        print(f"\nExperiment: {exp_dir.name}")
        summarize(denoising_steps, summary)


@cli.command()
@click.option(
    "--results",
    type=click.Path(exists=True),
    help="The path to the directory containing the results.",
)
@click.option(
    "--gt",
    type=click.Path(exists=True),
    help="The path to the directory containing the ground truth files.",
)
def avg_monomers_search(results: str, gt: str):
    """
    Compute average pLDDT, pTM, confidence score, and lDDT for each experiment.
    Separates results for random sampling and zero-order search.
    """

    def get_monomer_result(monomer_dir, search_method):
        pred_dir = (
            monomer_dir
            / pathlib.Path(f"{search_method}_{monomer_dir.name}")
            / "predictions"
            / monomer_dir.name
        )
        json_file = next(pred_dir.glob("*.json"))
        cif_file = next(pred_dir.glob("*.cif"))
        with open(json_file, "r") as f:
            data = json.load(f)
        return (
            float(data["complex_plddt"]),
            float(data["ptm"]),
            float(data["confidence_score"]),
            cif_file,
        )

    root = pathlib.Path(results)
    sub_dirs = sorted([
        d
        for d in root.iterdir()
        if d.is_dir() and d.name != "plots" and d.name != ".DS_Store"
    ])
    gt_files = sorted(
        [
            d
            for d in pathlib.Path(gt).iterdir()
            if not d.is_dir() and d.name != "plots" and d.name != ".DS_Store"
        ],
        key=lambda x: x.name,
    )

    first_monomer_dirs = sorted(
        [d for d in sub_dirs[0].iterdir() if d.is_dir() and d.name != "plots"],
        key=lambda x: x.name,
    )
    gt_files = gt_files[:len(first_monomer_dirs)]

    for sub_dir in tqdm(sub_dirs, desc="Processing experiments"):
        random_plddt, random_ptm, random_conf, random_lddt = [], [], [], []
        zos_plddt, zos_ptm, zos_conf, zos_lddt = [], [], [], []

        monomer_dirs = sorted(
            [d for d in sub_dir.iterdir() if d.is_dir() and d.name != "plots"],
            key=lambda x: x.name,
        )

        for monomer_dir, gt_cif in tqdm(
            zip(monomer_dirs, gt_files), desc="Processing monomers"
        ):
            assert (
                monomer_dir.name.split("_")[0] == gt_cif.stem
            ), f"Mismatch: {monomer_dir.name} vs {gt_cif.stem}"
            plddt, ptm, conf, random_cif = get_monomer_result(monomer_dir, "random")
            random_plddt.append(plddt)
            random_ptm.append(ptm)
            random_conf.append(conf)
            random_lddt.append(compute_lddt(random_cif, gt_cif))

            plddt, ptm, conf, zero_order_cif = get_monomer_result(
                monomer_dir, "zero_order"
            )
            zos_plddt.append(plddt)
            zos_ptm.append(ptm)
            zos_conf.append(conf)
            zos_lddt.append(compute_lddt(zero_order_cif, gt_cif))

        def summarize(search_method, plddt_list, ptm_list, conf_list, lddt_list):
            print(f"\n{search_method} Sampling")
            print("-" * 30)
            print(f"Avg pLDDT: {np.mean(plddt_list):.4f} ± {np.std(plddt_list):.4f}")
            print(f"Avg pTM: {np.mean(ptm_list):.4f} ± {np.std(ptm_list):.4f}")
            print(f"Avg Confidence: {np.mean(conf_list):.4f} ± {np.std(conf_list):.4f}")
            print(f"Avg LDDT: {np.mean(lddt_list):.4f} ± {np.std(lddt_list):.4f}")

        print("\n" + "=" * 50)
        print("Experiment: " + sub_dir.name)
        summarize("Random", random_plddt, random_ptm, random_conf, random_lddt)
        summarize("Zero-Order", zos_plddt, zos_ptm, zos_conf, zos_lddt)


@cli.command()
@click.argument("results_root", type=click.Path(exists=True))
def plot_single_sample_dist(results_root: str):
    """
    Plot pLDDT distributions for each sampling step experiment.
    """
    root = pathlib.Path(results_root)
    experiment_dirs = [
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("boltz_monomers")
    ]

    step_to_scores = {}

    for exp_dir in experiment_dirs:
        if exp_dir.name == "plots":
            continue
        try:
            name_parts = exp_dir.name.split("_")
            sampling_idx = name_parts.index("sampling")
            sampling_step = int(name_parts[sampling_idx + 1])
        except Exception:
            continue

        all_plddt = []

        for subdir in exp_dir.iterdir():
            pred_dir = subdir / "predictions"
            inner_dirs = [p for p in pred_dir.iterdir() if p.is_dir()]
            if not inner_dirs:
                continue
            pred_dir = inner_dirs[0]
            json_files = list(pred_dir.glob("*.json"))
            if not json_files:
                continue

            with open(json_files[0], "r") as f:
                data = json.load(f)
            all_plddt.append(float(data["complex_plddt"]))

        if all_plddt:
            step_to_scores[sampling_step] = all_plddt

    if not step_to_scores:
        print("No valid data found.")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    for step, scores in sorted(step_to_scores.items()):
        kde = gaussian_kde(scores)
        x_range = np.linspace(min(scores), max(scores), 1000)
        plt.plot(x_range, kde(x_range), label=f"{step} steps")

    plt.xlabel("pLDDT")
    plt.ylabel("Density")
    plt.title("pLDDT Distributions by Sampling Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plots_dir = root / "plots"
    plots_dir.mkdir(exist_ok=True)
    out_path = plots_dir / "plddt_distributions_by_sampling_step.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    cli()
