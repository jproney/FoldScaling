import pathlib
from dataclasses import asdict
from pathlib import Path
import shutil
from typing import Literal, Optional

import click
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

from boltz.utils import compute_lddt


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


def plddt_score(out, gt_cifs=None):
    """Calculate the pLDDT score from the output."""
    plddt = out["plddt"].mean(dim=1)
    return plddt.item()


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
    verifier: str = "plddt",
    gt_cifs: str = None,
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

    def lddt_score(out, gt_cifs=None):
        """Calculate the LDDT score from the output."""
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
            outputs=out,
            batch=batch,
            batch_idx=0,
            dataloader_idx=0,
        )

        pred_dir = out_dir / "predictions" / "_".join(out_dir.stem.split("_")[2:])
        cif_true = pathlib.Path(gt_cifs) / (out_dir.stem.split("_")[2] + ".cif")

        cif_pred = next(pred_dir.glob("*.cif"))
        assert cif_pred.exists(), f"Predicted cif file not found in {cif_pred}"
        assert cif_true.exists(), f"True cif file not found in {cif_true}"
        score, total = compute_lddt(cif_pred, cif_true)

        # remove temp dir
        shutil.rmtree(out_dir / "predictions")
        (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
        return score

    if verifier == "plddt":
        score_fn = plddt_score
    elif verifier == "lddt":
        score_fn = lddt_score
    else:
        raise ValueError(f"Unknown verifier: {verifier}. Use 'plddt' or 'lddt'.")

    # new code
    batch = next(iter(data_module.predict_dataloader()))
    device = model_module.device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

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
            score = score_fn(out, gt_cifs=gt_cifs)

            inner_iter.set_postfix(**{verifier: f"{score:.3f}"}, refresh=True)

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
    device="cuda",
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
        for sigma in np.linspace(
            sigma_start.cpu().item(), search_start_sigma, diffusion_solver_steps
        ):
            denoised_coords, _ = (
                model_module.structure_module.preconditioned_network_forward(
                    noised_atom_coords=noise,
                    sigma=sigma,
                    network_condition_kwargs=dict(feats=batch, multiplicity=1),
                    training=False,
                )
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
                for sigma_bwd in np.linspace(
                    sigma + forward_stepsize, sigma_backward_end, diffusion_solver_steps
                ):
                    denoised_coords, _ = (
                        model_module.structure_module.preconditioned_network_forward(
                            noise_bwd,
                            sigma_bwd,
                            dict(feats=batch, multiplicity=1),
                            training=False,
                        )
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
    verifier: str = "plddt",
    gt_cifs: str = None,
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

    def lddt_score(out, gt_cifs=None):
        """Calculate the LDDT score from the output."""
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
            outputs=out,
            batch=batch,
            batch_idx=0,
            dataloader_idx=0,
        )

        pred_dir = out_dir / "predictions" / "_".join(out_dir.stem.split("_")[1:])
        cif_true = pathlib.Path(gt_cifs) / (out_dir.stem.split("_")[1] + ".cif")
        
        cif_pred = next(pred_dir.glob("*.cif"))
        assert cif_pred.exists(), f"Predicted cif file not found in {cif_pred}"
        # cif_true = pathlib.Path(gt_cifs) / (out_dir.stem.split("_")[1] + ".cif")
        assert cif_true.exists(), f"True cif file not found in {cif_true}"
        score, total = compute_lddt(cif_pred, cif_true)

        # remove temp dir
        shutil.rmtree(out_dir / "predictions")
        (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
        return score

    if verifier == "plddt":
        score_fn = plddt_score
    elif verifier == "lddt":
        score_fn = lddt_score
    else:
        raise ValueError(f"Unknown verifier: {verifier}. Use 'plddt' or 'lddt'.")

    for i in tqdm(range(num_random_samples), desc="Random Sampling"):
        random_noise = torch.randn(noise_shape, device=device)
        model_module.custom_noise = random_noise
        out = model_module.predict_step(batch, batch_idx=0)
        score = score_fn(out, gt_cifs=gt_cifs)
        random_scores.append(score)
        click.echo(f"Sample {i+1}: Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_out = out

    click.echo("\nRandom Sampling Summary:")
    click.echo(f"Best Score: {best_score:.4f}")
    click.echo(f"Worst Score: {min(random_scores):.4f}")
    click.echo(f"Average Score: {np.mean(random_scores):.4f}")
    click.echo(
        f"Difference (Best - Worst): {(best_score - min(random_scores)):.4f}"
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
