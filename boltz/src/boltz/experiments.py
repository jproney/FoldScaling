import gc
import json
import pathlib
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Optional

import click
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import torch
import numpy as np
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
from typing import Callable

from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1
import shutil
from Bio.PDB.MMCIFParser import MMCIFParser
from scipy.spatial.distance import cdist

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


def plddt_score(out):
    """Calculate the pLDDT score from the output."""
    plddt = out["plddt"].mean(dim=1)
    return plddt


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
    score_fn: Callable = plddt_score,
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
            score = score_fn(out)
            score = score.item() if isinstance(score, torch.Tensor) else score

            inner_iter.set_postfix(**{score_fn.__name__: f"{score:.3f}"}, refresh=True)

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
    print("Best score:", best_score)

def search_over_paths(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    sampling_steps: int = 200,
    step_scale: float = 1.638,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    seed: Optional[int] = None,
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    num_initial_paths: int = 4,
    path_width: int = 5,
    search_start_sigma: float = 10.0,
    backward_stepsize: float = 0.5,
    forward_stepsize: float = 0.5,
    diffusion_solver_steps: int = 50,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
    no_potentials: bool = True,
    confidence_fk: bool = False,
    device="cuda"
):
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    if seed is not None:
        seed_everything(seed)

    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser() / f"search_over_paths_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
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
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
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
        map_location=device,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(PairformerArgs()),
        msa_module_args=asdict(MSAModuleArgs()),
        steering_args=asdict(steering_args),
    )
    model_module.eval()

    batch = next(iter(data_module.predict_dataloader()))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    atom_mask = batch["atom_pad_mask"]
    noise_shape = (*atom_mask.shape, 3)

    # Initialize N starting paths
    current_noises = torch.randn((num_initial_paths,) + noise_shape, device=device)
    sigmas = model_module.structure_module.sample_schedule(diffusion_solver_steps)
    sigma_start = sigmas[0].detach().cpu().item()


    for idx in range(num_initial_paths):
        noise = current_noises[idx]
        for sigma in np.linspace(sigma_start.cpu().item(), search_start_sigma, diffusion_solver_steps):
            denoised_coords, _ = model_module.structure_module.preconditioned_network_forward(
                noised_atom_coords=noise,
                sigma=sigma,
                network_condition_kwargs=dict(feats=batch, multiplicity=1),
                training=False
            )
            noise = denoised_coords
        current_noises[idx] = noise

    sigma = search_start_sigma

    # Iterative forward-noise and backward-denoise search
    while sigma > 0:
        candidate_noises, candidate_scores = [], []

        for candidate in current_noises:
            forward_noises = [
                candidate + torch.randn_like(candidate) * forward_stepsize
                for _ in range(path_width)
            ]

            backward_noises = []
            sigma_backward_end = max(sigma + forward_stepsize - backward_stepsize, 0)

            for noise_fwd in forward_noises:
                noise_bwd = noise_fwd
                for sigma_bwd in np.linspace(sigma + forward_stepsize, sigma_backward_end, diffusion_solver_steps):
                    denoised_coords, _ = model_module.structure_module.preconditioned_network_forward(
                        noise_bwd, sigma_bwd, dict(feats=batch, multiplicity=1), training=False
                    )
                    noise_bwd = denoised_coords
                backward_noises.append(noise_bwd)

            for noise_bwd in backward_noises:
                model_module.custom_noise = noise_bwd
                pred_out = model_module.predict_step(batch, batch_idx=0)
                score = pred_out["plddt"].mean().item()
                candidate_noises.append(noise_bwd)
                candidate_scores.append(score)

        top_indices = np.argsort(candidate_scores)[-num_initial_paths:]
        current_noises = torch.stack([candidate_noises[i] for i in top_indices])
        sigma = max(sigma - backward_stepsize, 0)

    best_idx = np.argmax(candidate_scores)
    best_noise = candidate_noises[best_idx]
    model_module.custom_noise = best_noise
    best_out = model_module.predict_step(batch, batch_idx=0)
    best_score = candidate_scores[best_idx]

    print("Final best PLDDT:", best_score)

    return best_out, best_score



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
    score_fn: Callable = plddt_score,
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
        score = score_fn(out)
        random_scores.append(score)
        click.echo(f"Sample {i+1}: Score = {score.item():.4f}")

        if score > best_score:
            best_score = score
            best_out = out

    click.echo("\nRandom Sampling Summary:")
    click.echo(f"Best Score: {best_score.item():.4f}")
    click.echo(f"Worst Score: {min(random_scores).item():.4f}")
    click.echo(f"Average Score: {np.mean([s.item() for s in random_scores]):.4f}")
    click.echo(
        f"Difference (Best - Worst): {(best_score - min(random_scores)).item():.4f}"
    )

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
@click.argument("results_root", type=click.Path(exists=True))
def table_single_sample(results_root: str):
    """
    Summarize average pLDDT and pTM for each sampling step experiment.
    """
    root = pathlib.Path(results_root)
    experiment_dirs = [
        d for d in root.iterdir() if d.is_dir() and d.name.startswith("boltz_monomers")
    ]
    summary = {}

    for exp_dir in experiment_dirs:
        if exp_dir.name == "plots":
            continue
        name_parts = exp_dir.name.split("_")
        sampling_idx = name_parts.index("sampling")
        sampling_step = int(name_parts[sampling_idx + 1])
        summary[sampling_step] = []
        all_plddt, all_ptm, all_conf = [], [], []

        for subdir in exp_dir.iterdir():
            pred_dir = subdir / "predictions"
            inner_dirs = [p for p in pred_dir.iterdir() if p.is_dir()]
            if not inner_dirs:
                print(f"No subdirectory inside {pred_dir}, skipping...")
                continue
            pred_dir = inner_dirs[0]
            json_file = next(pred_dir.glob("*.json"))
            with open(json_file, "r") as f:
                data = json.load(f)
            all_plddt.append(float(data["complex_plddt"]))
            all_ptm.append(float(data["ptm"]))
            all_conf.append(float(data["confidence_score"]))

        if all_plddt:
            avg_plddt = np.mean(all_plddt)
            avg_ptm = np.mean(all_ptm)
            avg_conf = np.mean(all_conf)
            summary[sampling_step] = (avg_plddt, avg_ptm, avg_conf)

    print("\nSampling Steps | Avg pLDDT | Avg pTM | Avg Confidence")
    print("-" * 55)
    for step in sorted(summary.keys()):
        if not summary[step]:
            continue
        plddt, ptm, conf = summary[step]
        print(f"{step:<14}   {plddt:.4f}     {ptm:.4f}    {conf:.4f}")


@cli.command()
@click.argument("results_root", type=click.Path(exists=True))
def table_monomers_predict(results_root: str):
    """
    Compute average pLDDT, pTM, and confidence score for all monomers.
    Separates results for random sampling and zero-order search.
    """
    root = pathlib.Path(results_root)
    random_plddt, random_ptm, random_conf = [], [], []
    zos_plddt, zos_ptm, zos_conf = [], [], []

    # sort directories by name
    monomer_dirs = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name != "plots"],
        key=lambda x: x.name,
    )
    monomer_dirs = monomer_dirs[:10] # take the first 10 directories

    for monomer_dir in monomer_dirs:
        if not monomer_dir.is_dir() or monomer_dir.name == "plots":
            continue

        # random
        pred_dir = monomer_dir / pathlib.Path("random_" + monomer_dir.name) / "predictions" / monomer_dir.name
        random_json = next(pred_dir.glob("*.json"))

        with open(random_json, "r") as f:
            data = json.load(f)
        random_plddt.append(float(data["complex_plddt"]))
        random_ptm.append(float(data["ptm"]))
        random_conf.append(float(data["confidence_score"]))

        # zero-order
        pred_dir = monomer_dir / pathlib.Path("zero_order_" + monomer_dir.name) / "predictions" / monomer_dir.name
        random_json = next(pred_dir.glob("*.json"))

        with open(random_json, "r") as f:
            data = json.load(f)
        zos_plddt.append(float(data["complex_plddt"]))
        zos_ptm.append(float(data["ptm"]))
        zos_conf.append(float(data["confidence_score"]))

    def summarize(name, plddt_list, ptm_list, conf_list):
        if not plddt_list:
            print(f"{name} — No data.")
            return
        avg_plddt = np.mean(plddt_list)
        avg_ptm = np.mean(ptm_list)
        avg_conf = np.mean(conf_list)
        print(f"{name:<12} | pLDDT: {avg_plddt:.4f} | pTM: {avg_ptm:.4f} | Confidence: {avg_conf:.4f}")

    print("\nSummary of Averages Across All Monomers")
    print("----------------------------------------")
    summarize("Random", random_plddt, random_ptm, random_conf)
    summarize("Zero-Order", zos_plddt, zos_ptm, zos_conf)


def extract_coords_from_cif(file_path):
    parser = MMCIFParser()
    structure = parser.get_structure('protein', file_path)

    residues = [res for res in structure.get_residues() if 'CA' in res]
    coords = np.array([res['CA'].get_coord() for res in residues])
    residue_ids = [res.get_id()[1] for res in residues]

    return coords, residue_ids

def align_coords(coords_pred, ids_pred, coords_true, ids_true):
    common_ids = sorted(set(ids_pred) & set(ids_true))
    idx_pred = [ids_pred.index(i) for i in common_ids]
    idx_true = [ids_true.index(i) for i in common_ids]
    return coords_pred[idx_pred], coords_true[idx_true]

def compute_lddt(cif_pred, cif_true, cutoff=15.0, per_atom=False):
    # Extract coordinates and residue IDs
    coords_pred, ids_pred = extract_coords_from_cif(cif_pred)
    coords_true, ids_true = extract_coords_from_cif(cif_true)

    # Align coordinates by residue IDs
    coords_pred_aligned, coords_true_aligned = align_coords(coords_pred, ids_pred, coords_true, ids_true)

    # Convert to distance matrices
    dmat_pred = torch.tensor(cdist(coords_pred_aligned, coords_pred_aligned))
    dmat_true = torch.tensor(cdist(coords_true_aligned, coords_true_aligned))

    # Compute mask (excluding self-distances)
    n_atoms = dmat_true.shape[0]
    mask = 1 - torch.eye(n_atoms)

    return lddt_dist(dmat_pred, dmat_true, mask, cutoff, per_atom)

# Provided lddt_dist function
def lddt_dist(dmat_predicted, dmat_true, mask, cutoff=15.0, per_atom=False):
    dists_to_score = (dmat_true < cutoff).float() * mask
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    score = 0.25 * (
        (dist_l1 < 0.5).float()
        + (dist_l1 < 1.0).float()
        + (dist_l1 < 2.0).float()
        + (dist_l1 < 4.0).float()
    )

    if per_atom:
        mask_no_match = torch.sum(dists_to_score, dim=-1) != 0
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=-1))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=-1))
        return score, mask_no_match.float()
    else:
        norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=(-2, -1)))
        score = norm * (1e-10 + torch.sum(dists_to_score * score, dim=(-2, -1)))
        total = torch.sum(dists_to_score, dim=(-1, -2))
        return score.item(), total.item()


@cli.command()
@click.option(
    "--results_dir",
    type=click.Path(exists=True),
    help="The path to the directory containing the fasta files.",
)
@click.option(
    "--gt_dir",
    type=click.Path(exists=True),
    help="The path to the directory containing the ground truth files.",
)
def run_lddt(results_dir: str, gt_dir: str):
    """
    Assumes that the number of monomers are the same in both directories.
    """
    root_dir = pathlib.Path(results_dir)
    gt_dir = pathlib.Path(gt_dir)

    # sort directories by name
    monomer_dirs = sorted(
        [d for d in root_dir.iterdir() if d.is_dir() and d.name != "plots"],
        key=lambda x: x.name,
    )
    print("Monomer directories: " + str([monomer.name.split("_")[0] for monomer in monomer_dirs]))

    gt_dirs = sorted(
        [d for d in gt_dir.iterdir() if d.name != "plots" and d.name != ".DS_Store"],
        key=lambda x: x.name,
    )
    print("Ground truth directories: " + str([monomer.name for monomer in gt_dirs]))

    for monomer_dir, monomer in zip(monomer_dirs, gt_dirs):
        if not monomer_dir.is_dir() or monomer_dir.name == "plots":
            continue

        # random
        pred_dir = monomer_dir / pathlib.Path("random_" + monomer_dir.name) / "predictions" / monomer_dir.name
        random_cif = next(pred_dir.glob("*.cif"))

        random_json = next(pred_dir.glob("*.json"))
        with open(random_json, "r") as f:
            data = json.load(f)
        random_plddt = float(data["complex_plddt"])

        # zero-order
        pred_dir = monomer_dir / pathlib.Path("zero_order_" + monomer_dir.name) / "predictions" / monomer_dir.name
        zero_order_cif = next(pred_dir.glob("*.cif"))

        zero_order_json = next(pred_dir.glob("*.json"))
        with open(zero_order_json, "r") as f:
            data = json.load(f)
        zero_order_plddt = float(data["complex_plddt"])

        print(f"\n------\nProcessing {monomer_dir.name},{monomer.name}")

        # Compute LDDT
        random_lddt = compute_lddt(random_cif, monomer)
        zero_order_lddt = compute_lddt(zero_order_cif, monomer)

        print(f"Random LDDT for {monomer_dir.name}: {random_lddt}")
        print(f"Random pLDDT for {monomer_dir.name}: {random_plddt}")

        print(f"Zero-Order LDDT for {monomer_dir.name}: {zero_order_lddt}")
        print(f"Zero-Order pLDDT for {monomer_dir.name}: {zero_order_plddt}")


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
