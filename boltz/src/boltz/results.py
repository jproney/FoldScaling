import json
import pathlib
import re
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import click
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

from boltz.utils import compute_lddt

warnings.simplefilter("ignore", PDBConstructionWarning)


@click.group()
def cli() -> None:
    """Results with Boltz-1."""
    return


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
        print(f"Avg LDDT: {avg_lddt:.4f} ± {std_lddt:.4f}")

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
    gt_files = gt_files[: len(sub_dirs)]
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
    sub_dirs = sorted(
        [
            d
            for d in root.iterdir()
            if d.is_dir() and d.name != "plots" and d.name != ".DS_Store"
        ]
    )
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
    gt_files = gt_files[: len(first_monomer_dirs)]

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
            random_lddt.append(compute_lddt(random_cif, gt_cif)[0])

            plddt, ptm, conf, zero_order_cif = get_monomer_result(
                monomer_dir, "zero_order"
            )
            zos_plddt.append(plddt)
            zos_ptm.append(ptm)
            zos_conf.append(conf)
            zos_lddt.append(compute_lddt(zero_order_cif, gt_cif)[0])

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
@click.option(
    "--results",
    type=click.Path(exists=True),
    required=True,
    help="The path to the directory containing all experiment results.",
)
def plot_plddt_vs_nfe(results):
    """
    Generate line plots of average pLDDT vs. NFE for both random sampling
    and zero-order search across all experiments in the results directory.
    """

    def extract_config_from_name(name):
        match = re.search(
            r"denoising_(\d+)_recycling_\d+_random_samples_(\d+)_neighbors_(\d+)_iterations_(\d+)",
            name,
        )
        if not match:
            raise ValueError(f"Could not parse experiment name: {name}")
        denoising = int(match.group(1))
        samples = int(match.group(2))
        neighbors = int(match.group(3))
        iterations = int(match.group(4))
        return denoising, samples, neighbors, iterations

    def gather_plddt_scores(results_root):
        results_root = pathlib.Path(results_root)
        exp_dirs = sorted(
            [
                d
                for d in results_root.iterdir()
                if d.is_dir() and d.name.startswith("boltz_monomers")
            ]
        )

        random_data = []
        zos_data = []

        for exp in exp_dirs:
            try:
                denoising, samples, neighbors, iterations = extract_config_from_name(
                    exp.name
                )
            except ValueError:
                continue  # Skip unrecognized folder

            nfe_random = denoising * samples
            nfe_zos = denoising * neighbors * iterations

            monomer_dirs = sorted(
                [d for d in exp.iterdir() if d.is_dir() and d.name != "plots"]
            )
            if not monomer_dirs:
                continue

            plddts_random = []
            plddts_zos = []

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
                        continue  # Skip missing/incomplete predictions

            if plddts_random:
                avg_random = sum(plddts_random) / len(plddts_random)
                random_data.append((nfe_random, avg_random))
            if plddts_zos:
                avg_zos = sum(plddts_zos) / len(plddts_zos)
                zos_data.append((nfe_zos, avg_zos))

        return random_data, zos_data

    random_data, zos_data = gather_plddt_scores(results)

    random_data.sort()
    zos_data.sort()

    plt.figure()
    if random_data:
        x_r, y_r = zip(*random_data)
        plt.plot(x_r, y_r, marker="o", label="Random Sampling")
    if zos_data:
        x_z, y_z = zip(*zos_data)
        plt.plot(x_z, y_z, marker="o", label="Zero-Order Search")

    plt.xlabel("Number of Function Evaluations (NFE)")
    plt.ylabel("Average pLDDT")
    plt.title("pLDDT vs NFE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cli()
