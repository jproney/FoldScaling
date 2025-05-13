import json
import pathlib
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

import click
from matplotlib import pyplot as plt
import torch
import numpy as np
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm

from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1

from boltz.main import (
    BoltzConfidencePotential,
    BoltzDiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgs,
    download,
    check_inputs,
    process_inputs,
    BoltzProcessedInput,
)


def generate_neighbors(x, threshold=0.95, num_neighbors=4):
    """Courtesy: Willis Ma"""
    rng = np.random.Generator(np.random.PCG64())
    x_f = x.flatten(1)
    x_norm = torch.linalg.norm(
        x_f, dim=-1, keepdim=True, dtype=torch.float64
    ).unsqueeze(-2)
    u = x_f.unsqueeze(-2) / x_norm.clamp_min(1e-12)
    v = torch.from_numpy(
        rng.standard_normal(
            size=(u.shape[0], num_neighbors, u.shape[-1]), dtype=np.float64
        )
    ).to(u.device)
    w = F.normalize(v - (v @ u.transpose(-2, -1)) * u, dim=-1)
    return (
        (x_norm * (threshold * u + np.sqrt(1 - threshold**2) * w))
        .reshape(x.shape[0], num_neighbors, *x.shape[1:])
        .to(x.dtype)
    )


def generate_protein_neighbors(base_noise, threshold=0.95, num_neighbors=5):
    B, M, C = base_noise.shape
    flattened_dim = M * C

    # Flatten the base noise to (B, M*C)
    base_flat = base_noise.view(B, flattened_dim)  # [B, D], D=M*C

    # Generate random Gaussian noise
    random_noise = torch.randn(
        num_neighbors, B, flattened_dim, device=base_noise.device
    )

    # Project onto sphere defined by cosine similarity threshold
    base_norm = F.normalize(base_flat, dim=-1)  # [B, D]

    neighbors = []
    for i in range(num_neighbors):
        random_noise = torch.randn(B, flattened_dim, device=base_noise.device)

        proj = (random_noise * base_norm).sum(dim=-1, keepdim=True) * base_norm
        noise_orthogonal = random_noise - proj
        noise_orthogonal_normed = F.normalize(noise_orthogonal, dim=-1)

        neighbor_flat = (
            threshold * base_norm
            + torch.sqrt(torch.tensor(1 - threshold**2, device=base_noise.device))
            * noise_orthogonal_normed
        )
        neighbor_flat_scaled = neighbor_flat * base_flat.norm(dim=-1, keepdim=True)

        neighbor = neighbor_flat_scaled.view(B, M, C)
        neighbors.append(neighbor)
    return torch.stack(neighbors)


def zero_order_sampling(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    sampling_steps: int = 200,
    step_scale: float = 1.638,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    no_potentials: bool = False,
    num_candidates: int = 8,
    recycling_steps: int = 3,
    num_iterations: int = 10,
    confidence_fk: bool = False,
    diffusion_samples: int = 1,
) -> None:
    """Run predictions with Boltz-1."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    # Set cache path
    cache = pathlib.Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = pathlib.Path(data).expanduser()
    out_dir = pathlib.Path(out_dir).expanduser()
    out_dir = out_dir / f"zero_order_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache)

    # Validate inputs
    data = check_inputs(data, out_dir, override)
    if not data:
        click.echo("No predictions to run, exiting.")
        return

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()
        if len(data) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions."
            )
            raise ValueError(msg)

    msg = f"Running predictions for {len(data)} structure"
    msg += "s" if len(data) > 1 else ""
    click.echo(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
        constraints_dir=processed.constraints_dir,
    )

    # Load model
    if checkpoint is None:
        checkpoint = cache / "boltz1_conf.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = step_scale

    pairformer_args = PairformerArgs()
    msa_module_args = MSAModuleArgs()

    steering_args = BoltzSteeringParams()
    if no_potentials:
        steering_args.fk_steering = False
        steering_args.guidance_update = False

    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cuda" if accelerator == "gpu" else "cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(pairformer_args),
        msa_module_args=asdict(msa_module_args),
        steering_args=asdict(steering_args),
    )
    model_module.eval()

    if confidence_fk:
        model_module.steering_args["num_particles"] = 8
        model_module.steering_args["fk_lambda"] = 50
        model_module.steering_args["fk_resampling_interval"] = 5
        model_module.steering_args["guidance_update"] = False
        model_module.steering_args["max_fk_noise"] = 100
        model_module.steering_args["potential_type"] = "vanilla"
        model_module.steering_args["noise_coord_potential"] = False

        pot = BoltzConfidencePotential(
            parameters={
                "guidance_interval": 5,
                "guidance_weight": 0.00,
                "resampling_weight": 1.0,
                "model": model_module,
                "total_particles": diffusion_samples
                * model_module.steering_args["num_particles"],
            }
        )

        model_module.predict_args["confidence_potential"] = pot
        model_module.confidence_module.use_s_diffusion = False
    else:
        model_module.predict_args["confidence_potential"] = None

    # new code
    batch = next(iter(data_module.predict_dataloader()))
    device = model_module.device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # print("hi", batch["atom_pad_mask"].shape)
    atom_mask = batch["atom_pad_mask"]
    noise_shape = (*atom_mask.shape, 3)

    base_noise = torch.randn(noise_shape, device=atom_mask.device)
    best_score = -float("inf")
    best_out = None

    for iteration in tqdm(range(num_iterations), desc="Zero-order optimization"):
        print(f"\n--- Starting Iteration {iteration} ---")
        top_candidate_noise = None
        top_score = -float("inf")

        neighbors = generate_protein_neighbors(base_noise, num_neighbors=num_candidates)

        previous_best_score = best_score  # Track the previous best score explicitly

        inner_iter = tqdm(
            range(num_candidates), desc=f"Iteration {iteration}", leave=False
        )
        for i in inner_iter:
            candidate_noise = neighbors[i]

            # Set custom noise
            model_module.custom_noise = candidate_noise
            out = model_module.predict_step(batch, batch_idx=0)
            score = out["plddt"].mean().item()

            inner_iter.set_postfix(PLDDT=f"{score:.3f}", refresh=True)

            # Find best candidate this iteration
            if score > top_score:
                top_score = score
                top_candidate_noise = candidate_noise

            # Track global best result
            if score > best_score:
                best_score = score
                best_out = out

        print(f"Iteration {iteration} summary:")
        print(f" - Best score this iteration: {top_score:.4f}")
        print(f" - Previous global best score: {previous_best_score:.4f}")

        # Check for improvement compared to previous global best
        if top_score > previous_best_score:
            base_noise = top_candidate_noise
            print(f"Iteration {iteration}: Improvement found, updating base noise.")
        else:
            base_noise = torch.randn(noise_shape, device=atom_mask.device)
            print(f"Iteration {iteration}: No improvement, regenerating base noise.")

    # Final summary
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
    )
    trainer = Trainer(
        default_root_dir=out_dir,
        strategy="auto",
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )
    pred_writer.on_predict_batch_end(
        trainer=trainer,
        pl_module=model_module,
        outputs=best_out,
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )
    print("Best PLDDT score:", best_score)


def random_sampling(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    sampling_steps: int = 200,
    step_scale: float = 1.638,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    no_potentials: bool = False,
    num_candidates: int = 8,
    num_random_samples: int = 10,
    recycling_steps: int = 3,
    confidence_fk: bool = False,
    diffusion_samples: int = 1,
) -> None:
    """Run random sampling predictions with Boltz-1."""
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    if seed is not None:
        seed_everything(seed)

    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser() / f"random_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    download(cache)

    data = check_inputs(data, out_dir)
    if not data:
        click.echo("No predictions to run, exiting.")
        return

    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        use_msa_server=use_msa_server,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
    )

    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
    )

    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
        constraints_dir=processed.constraints_dir,
    )

    if checkpoint is None:
        checkpoint = cache / "boltz1_conf.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
    }
    diffusion_params = BoltzDiffusionParams(step_scale=step_scale)

    steering_args = BoltzSteeringParams()
    if no_potentials:
        steering_args.fk_steering = False
        steering_args.guidance_update = False

    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cuda" if accelerator == "gpu" else "cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(PairformerArgs()),
        msa_module_args=asdict(MSAModuleArgs()),
        steering_args=asdict(steering_args),
    )
    model_module.eval()

    if confidence_fk:
        model_module.steering_args["num_particles"] = 8
        model_module.steering_args["fk_lambda"] = 50
        model_module.steering_args["fk_resampling_interval"] = 5
        model_module.steering_args["guidance_update"] = False
        model_module.steering_args["max_fk_noise"] = 100
        model_module.steering_args["potential_type"] = "vanilla"
        model_module.steering_args["noise_coord_potential"] = False

        pot = BoltzConfidencePotential(
            parameters={
                "guidance_interval": 5,
                "guidance_weight": 0.00,
                "resampling_weight": 1.0,
                "model": model_module,
                "total_particles": diffusion_samples
                * model_module.steering_args["num_particles"],
            }
        )

        model_module.predict_args["confidence_potential"] = pot
        model_module.confidence_module.use_s_diffusion = False
    else:
        model_module.predict_args["confidence_potential"] = None

    batch = next(iter(data_module.predict_dataloader()))
    device = model_module.device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    atom_mask = batch["atom_pad_mask"]
    noise_shape = (*atom_mask.shape, 3)

    random_scores = []
    best_score, best_out = -float("inf"), None

    for i in tqdm(range(num_random_samples), desc="Random Sampling"):
        random_noise = torch.randn(noise_shape, device=device)
        model_module.custom_noise = random_noise
        out = model_module.predict_step(batch, batch_idx=0)
        score = out["plddt"].mean().item()
        random_scores.append(score)
        click.echo(f"Sample {i+1}: PLDDT = {score:.4f}")

        if score > best_score:
            best_score = score
            best_out = out

    click.echo("\nRandom Sampling Summary:")
    click.echo(f"Best PLDDT: {best_score:.4f}")
    click.echo(f"Worst PLDDT: {min(random_scores):.4f}")
    click.echo(f"Average PLDDT: {np.mean(random_scores):.4f}")
    click.echo(f"Difference (Best - Worst): {best_score - min(random_scores):.4f}")

    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
    )
    trainer = Trainer(
        default_root_dir=out_dir,
        strategy="auto",
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )
    pred_writer.on_predict_batch_end(
        trainer=trainer,
        pl_module=model_module,
        outputs=best_out,
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )
    click.echo(f"Best structure written. PLDDT: {best_score:.4f}")


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
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction.",
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
def monomers_predict(
    data_dir: str,
    use_msa: bool,
    sampling_steps: int,
    recycling_steps: int,
    num_random_samples: int,
    num_neighbors: int,
    num_iterations: int,
) -> None:
    """Make sure to run this command inside the data directory."""

    parent_dir = pathlib.Path(data_dir).absolute().parent.parent
    results_dir = parent_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_dir = (
        results_dir
        / f"boltz_monomers_msa_{use_msa}_sampling_{sampling_steps}_recycling_{recycling_steps}_random_samples_{num_random_samples}_neighbors_{num_neighbors}_iterations_{num_iterations}"
    )
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fasta_files = list(pathlib.Path(data_dir).glob("*.fasta"))

    if use_msa:
        fasta_files = [f for f in all_fasta_files if "_no_msa" not in f.name]
    else:
        fasta_files = [f for f in all_fasta_files if "_no_msa" in f.name]

    for fasta in tqdm(fasta_files, desc="Processing monomers"):
        print(f"\n------\nProcessing {fasta.name}")

        # make a new directory for each fasta file
        fasta_name = fasta.stem
        sub_dir = out_dir / fasta_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        random_sampling(
            data=str(fasta),
            out_dir=str(sub_dir),
            devices=1,
            accelerator="gpu",
            sampling_steps=sampling_steps,
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
        )

        zero_order_sampling(
            data=str(fasta),
            out_dir=str(sub_dir),
            devices=1,
            accelerator="gpu",
            sampling_steps=sampling_steps,
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
        )


@cli.command()
@click.argument("results_root", type=click.Path(exists=True))
def plot_plddt_diffs(results_root: str):
    root = pathlib.Path(results_root)
    save_path = root / "plots"
    save_path.mkdir(parents=True, exist_ok=True)
    differences = []

    subdirs = [d for d in root.iterdir() if d.is_dir() and d.name != "plots"]

    for subdir in tqdm(subdirs, desc="Comparing ZOS vs Random"):
        # find random directory
        random_dir = next(subdir.glob("random_*"), None)
        random_predictions = random_dir / "predictions" / subdir.name
        random_results = next(random_predictions.glob("*.json"), None)

        # find zero-order directory
        zero_order_dir = next(subdir.glob("zero_order_*"), None)
        zero_order_predictions = zero_order_dir / "predictions" / subdir.name
        zero_order_results = next(zero_order_predictions.glob("*.json"), None)

        # Load scores
        with open(random_results, "r") as f:
            random_data = json.load(f)
        with open(zero_order_results, "r") as f:
            zos_data = json.load(f)

        diff = zos_data["complex_plddt"] - random_data["complex_plddt"]
        differences.append(diff)

    # Plot and save
    plt.figure(figsize=(8, 5))
    plt.hist(differences, bins=20, color="skyblue", edgecolor="black")
    plt.axvline(0, color="red", linestyle="--")
    plt.title("PLDDT Difference per Monomer (ZOS - Random)")
    plt.xlabel("Difference in Best PLDDT")
    plt.ylabel("Number of Monomers")
    plt.tight_layout()

    plot_file = save_path / "plddt_diff_histogram.png"
    plt.savefig(plot_file)
    print(f"Saved histogram to: {plot_file}")


if __name__ == "__main__":
    cli()
