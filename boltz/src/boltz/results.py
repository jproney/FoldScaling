import json
import math
import pathlib
import re
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import click
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from boltz.utils import compute_lddt
import seaborn as sns

warnings.simplefilter("ignore", PDBConstructionWarning)


@click.group()
def cli() -> None:
    """Results with Boltz-1."""
    return


@cli.command()
@click.option(
    "--results",
    type=click.Path(exists=True),
    required=True,
    help="Path to a parent directory containing `search_results/` and `denoising_results/`.",
)
@click.option(
    "--num_decimals",
    type=int,
    default=3,
)
@click.option(
    "--num_monomers",
    type=int,
    default=25,
)
@click.option(
    "--no_show_single",
    is_flag=True,
    default=False,
)
def plot_nfe_vs_plddt(results, num_decimals, num_monomers, no_show_single):
    """
    Plot average pLDDT vs NFE for:
    - Random Sampling
    - Zero-Order Search
    - Single-sample runs (varying denoising)
    """
    results = pathlib.Path(results)
    search_results = results / "search_results_plddt"
    denoising_results = results / "denoising_results"

    def extract_config_from_name(name):
        match = re.search(
            r"denoising_(\d+)_recycling_\d+_random_samples_(\d+)_neighbors_(\d+)_iterations_(\d+)",
            name,
        )
        if not match:
            raise ValueError(f"Could not parse experiment name: {name}")
        return map(int, match.groups())

    def gather_search_data(search_root):
        random_data = []
        zos_data = []
        scores_by_nfe_random = {}
        scores_by_nfe_zos = {}

        for exp in sorted(search_root.glob("boltz_monomers*")):
            try:
                denoising, samples, neighbors, iterations = extract_config_from_name(
                    exp.name
                )
            except ValueError:
                continue

            nfe_random = denoising * samples
            nfe_zos = denoising * neighbors * iterations

            monomer_dirs = sorted(
                [d for d in exp.iterdir() if d.is_dir() and d.name != "plots"]
            )
            monomer_dirs = monomer_dirs[:num_monomers]  # just to make sure
            plddts_random, plddts_zos = [], []

            for monomer_dir in monomer_dirs:
                for method in ["random", "zero_order"]:
                    pred_dir = (
                        monomer_dir
                        / f"{method}_{monomer_dir.name}"
                        / "predictions"
                        / monomer_dir.name
                    )
                    try:
                        json_file = next(pred_dir.glob("*.json"))
                        with open(json_file, "r") as f:
                            data = json.load(f)
                        if method == "random":
                            plddts_random.append(float(data["complex_plddt"]))
                        else:
                            plddts_zos.append(float(data["complex_plddt"]))
                    except Exception:
                        continue

            if plddts_random:
                random_data.append(
                    (nfe_random, np.mean(plddts_random), np.std(plddts_random))
                )
                scores_by_nfe_random.setdefault(nfe_random, []).extend(plddts_random)

            if plddts_zos:
                zos_data.append((nfe_zos, np.mean(plddts_zos), np.std(plddts_zos)))
                scores_by_nfe_zos.setdefault(nfe_zos, []).extend(plddts_zos)

        return random_data, zos_data, scores_by_nfe_random, scores_by_nfe_zos

    def gather_denoising_data(denoising_root):
        data = []
        scores_by_nfe_single = {}

        for exp in sorted(denoising_root.glob("boltz_monomers*")):
            match = re.search(r"denoising_(\d+)", exp.name)
            if not match:
                continue
            denoising = int(match.group(1))
            nfe = denoising  # only 1 sample

            monomer_dirs = sorted(
                [
                    d
                    for d in exp.iterdir()
                    if d.is_dir() and d.name.startswith("random_")
                ]
            )
            monomer_dirs = monomer_dirs[:num_monomers]  # just to make sure
            plddts = []

            for monomer_dir in monomer_dirs:
                pred_root = monomer_dir / "predictions"
                if not pred_root.exists():
                    continue

                inner_dirs = [d for d in pred_root.iterdir() if d.is_dir()]
                for inner in inner_dirs:
                    json_files = list(inner.glob("*.json"))
                    if not json_files or json_files[0].stat().st_size == 0:
                        continue
                    try:
                        with open(json_files[0], "r") as f:
                            data_json = json.load(f)
                        plddt = float(data_json["complex_plddt"])
                        plddts.append(plddt)
                    except Exception:
                        continue

            if plddts:
                data.append((nfe, np.mean(plddts), np.std(plddts)))
                scores_by_nfe_single.setdefault(nfe, []).extend(plddts)

        return data, scores_by_nfe_single

    # Gather data
    random_data, zos_data, scores_by_nfe_random, scores_by_nfe_zos = gather_search_data(search_results)
    single_sample_data, scores_by_nfe_single = gather_denoising_data(denoising_results)

    # Sort
    random_data.sort()
    zos_data.sort()
    single_sample_data.sort()

    # Function to create a row of (mean, std) tuples
    def tuple_row(data):
        return {nfe: (mean, std) for nfe, mean, std in data}

    # Create main DataFrame with tuples
    df = pd.DataFrame.from_dict(
        {
            "single sample": tuple_row(single_sample_data),
            "random": tuple_row(random_data),
            "zero order": tuple_row(zos_data),
        },
        orient="index",
    )

    # Format as "mean ± std" strings (preserve decimals)
    df_formatted = df.map(
        lambda x: f"{x[0]:.{num_decimals}f} ± {x[1]:.{num_decimals}f}"
    )

    # Print
    print(df_formatted.to_string())

    # Plot all
    plt.figure()
    if random_data:
        x_r, y_r, _ = zip(*random_data)
        plt.plot(x_r, y_r, marker="o", label="Random Sampling")
    if zos_data:
        x_z, y_z, _ = zip(*zos_data)
        plt.plot(x_z, y_z, marker="o", label="Zero-Order Search")
    if single_sample_data and not no_show_single:
        x_s, y_s, _ = zip(*single_sample_data)
        plt.plot(x_s, y_s, marker="o", label="Single Sample (Denoising Sweep)")

    plt.xlabel("Number of Function Evaluations (NFE)")
    plt.ylabel("Average pLDDT")
    plt.title("Average pLDDT vs NFE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- pLDDT Distributions by NFE ---
    def flatten_scores_by_nfe(score_dict, method_label):
        """
        Converts {nfe: [scores]} to list of dicts: [{"score": ..., "nfe": ..., "method": ...}, ...]
        """
        rows = []
        for nfe, scores in score_dict.items():
            rows.extend(
                {"score": s, "nfe": nfe, "method": method_label}
                for s in scores
                if not math.isnan(s)
            )
        return rows

    # Combine all data
    distribution_data = (
        flatten_scores_by_nfe(scores_by_nfe_random, "Random Sampling")
        + flatten_scores_by_nfe(scores_by_nfe_zos, "Zero-Order Search")
        + flatten_scores_by_nfe(scores_by_nfe_single, "Single Sample")
    )
    df_dist = pd.DataFrame(distribution_data)

    for method in df_dist["method"].unique():
        plt.figure()
        subset = df_dist[df_dist["method"] == method]
        sns.kdeplot(
            data=subset,
            x="score",
            hue="nfe",
            fill=False,
            common_norm=False,
            alpha=0.5,
            palette="tab10",
        )
        plt.title(f"pLDDT Distribution per NFE - {method}")
        plt.xlabel("pLDDT Score")
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


@cli.command()
@click.option(
    "--results",
    type=click.Path(exists=True),
    required=True,
    help="Path to a parent directory containing `search_results/` and `denoising_results/`.",
)
@click.option(
    "--num_decimals",
    type=int,
    default=3,
)
@click.option(
    "--num_monomers",
    type=int,
    default=25,
)
@click.option(
    "--gt",
    type=click.Path(exists=True),
    help="The path to the directory containing the ground truth files.",
)
def plot_nfe_vs_lddt(results, num_decimals: int, num_monomers: int, gt: str):
    """
    Plot average LDDT vs NFE for:
    - Random Sampling
    - Zero-Order Search
    - Single-sample runs (varying denoising)
    """
    results = pathlib.Path(results)
    search_results = results / "search_results_lddt"
    denoising_results = results / "denoising_results"
    cifs_true = sorted(
        [
            d
            for d in pathlib.Path(gt).iterdir()
            if not d.is_dir() and d.name != ".DS_Store"
        ]
    )
    cifs_true = cifs_true[:num_monomers]

    def extract_config_from_name(name):
        match = re.search(
            r"denoising_(\d+)_recycling_\d+_random_samples_(\d+)_neighbors_(\d+)_iterations_(\d+)",
            name,
        )
        if not match:
            raise ValueError(f"Could not parse experiment name: {name}")
        return map(int, match.groups())

    def gather_search_data(search_root):
        random_data = []
        zos_data = []
        scores_by_nfe_random = {}
        scores_by_nfe_zos = {}

        for exp in sorted(search_root.glob("boltz_monomers*")):
            try:
                denoising, samples, neighbors, iterations = extract_config_from_name(
                    exp.name
                )
            except ValueError:
                continue

            nfe_random = denoising * samples
            nfe_zos = denoising * neighbors * iterations

            monomer_dirs = sorted(
                [d for d in exp.iterdir() if d.is_dir() and d.name != "plots"]
            )
            monomer_dirs = monomer_dirs[:num_monomers]  # just to make sure
            lddts_random, lddts_zos = [], []

            for monomer_dir, cif_true in zip(monomer_dirs, cifs_true):
                for method in ["random", "zero_order"]:
                    pred_dir = (
                        monomer_dir
                        / f"{method}_{monomer_dir.name}"
                        / "predictions"
                        / monomer_dir.name
                    )
                    cif_pred = next(pred_dir.glob("*.cif"))
                    score, total = compute_lddt(cif_pred, cif_true)
                    if method == "random":
                        lddts_random.append(score)
                    else:
                        lddts_zos.append(score)

            if lddts_random:
                random_data.append(
                    (nfe_random, np.mean(lddts_random), np.std(lddts_random))
                )
                scores_by_nfe_random.setdefault(nfe_random, []).extend(lddts_random)
            if lddts_zos:
                zos_data.append((nfe_zos, np.mean(lddts_zos), np.std(lddts_zos)))
                scores_by_nfe_zos.setdefault(nfe_zos, []).extend(lddts_zos)

        return random_data, zos_data, scores_by_nfe_random, scores_by_nfe_zos

    def gather_denoising_data(denoising_root):
        data = []
        scores_by_nfe_single = {}

        for exp in sorted(denoising_root.glob("boltz_monomers*")):
            match = re.search(r"denoising_(\d+)", exp.name)
            if not match:
                continue
            denoising = int(match.group(1))
            nfe = denoising  # only 1 sample

            monomer_dirs = sorted(
                [
                    d
                    for d in exp.iterdir()
                    if d.is_dir() and d.name.startswith("random_")
                ]
            )
            monomer_dirs = monomer_dirs[:num_monomers]  # just to make sure
            lddts = []

            for monomer_dir, cif_true in zip(monomer_dirs, cifs_true):
                pred_root = monomer_dir / "predictions"
                if not pred_root.exists():
                    continue
                inner_dirs = [d for d in pred_root.iterdir() if d.is_dir()]
                for inner in inner_dirs:
                    cif_files = list(inner.glob("*.cif"))
                    if not cif_files or cif_files[0].stat().st_size == 0:
                        continue
                    score, total = compute_lddt(cif_files[0], cif_true)
                    lddts.append(score)

            if lddts:
                data.append((nfe, np.mean(lddts), np.std(lddts)))
                scores_by_nfe_single.setdefault(nfe, []).extend(lddts)

        return data, scores_by_nfe_single

    # Gather data
    random_data, zos_data, scores_by_nfe_random, scores_by_nfe_zos = gather_search_data(search_results)
    single_sample_data, scores_by_nfe_single = gather_denoising_data(denoising_results)

    # Sort
    random_data.sort()
    zos_data.sort()
    single_sample_data.sort()

    # Function to create a row of (mean, std) tuples
    def tuple_row(data):
        return {nfe: (mean, std) for nfe, mean, std in data}

    # Create main DataFrame with tuples
    df = pd.DataFrame.from_dict(
        {
            "single sample": tuple_row(single_sample_data),
            "random": tuple_row(random_data),
            "zero order": tuple_row(zos_data),
        },
        orient="index",
    )

    df_formatted = df.map(
    lambda x: f"{x[0]:.{num_decimals}f} ± {x[1]:.{num_decimals}f}"
    if isinstance(x, tuple) and all(isinstance(v, (int, float)) and not math.isnan(v) for v in x)
    else "-"
    )
    # df_formatted = df.map(
    #     lambda x: f"{x[0]:.{num_decimals}f} ± {x[1]:.{num_decimals}f}"
    # )

    # Print
    print(df_formatted.to_string())

    # Plot all
    plt.figure()
    if random_data:
        x_r, y_r, _ = zip(*random_data)
        plt.plot(x_r, y_r, marker="o", label="Random Sampling")
    if zos_data:
        x_z, y_z, _ = zip(*zos_data)
        plt.plot(x_z, y_z, marker="o", label="Zero-Order Search")
    if single_sample_data:
        x_s, y_s, _ = zip(*single_sample_data)
        plt.plot(x_s, y_s, marker="o", label="Single Sample (Denoising Sweep)")

    plt.xlabel("Number of Function Evaluations (NFE)")
    plt.ylabel("Average LDDT")
    plt.title("Average LDDT vs NFE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- LDDT Distributions by NFE ---
    def flatten_scores_by_nfe(score_dict, method_label):
        """
        Converts {nfe: [scores]} to list of dicts: [{"score": ..., "nfe": ..., "method": ...}, ...]
        """
        rows = []
        for nfe, scores in score_dict.items():
            rows.extend(
                {"score": s, "nfe": nfe, "method": method_label}
                for s in scores
                if not math.isnan(s)
            )
        return rows

    # Combine all data
    distribution_data = (
        flatten_scores_by_nfe(scores_by_nfe_random, "Random Sampling")
        + flatten_scores_by_nfe(scores_by_nfe_zos, "Zero-Order Search")
        + flatten_scores_by_nfe(scores_by_nfe_single, "Single Sample")
    )
    df_dist = pd.DataFrame(distribution_data)

    for method in df_dist["method"].unique():
        plt.figure()
        subset = df_dist[df_dist["method"] == method]
        sns.kdeplot(
            x=subset["score"].to_numpy(),  # <- workaround for pandas/seaborn bug
            hue=subset["nfe"],
            fill=False,
            common_norm=False,
            alpha=0.5,
            palette="tab10",
        )
        plt.title(f"LDDT Distribution per NFE - {method}")
        plt.xlabel("LDDT Score")
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    cli()
