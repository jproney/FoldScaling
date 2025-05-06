import os
import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import click
import torch
import numpy as np
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = (
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
)

# Add global variable to store predictions
_boltz_predictions = None


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path
    constraints_dir: Optional[Path] = None


@dataclass
class PairformerArgs:
    """Pairformer arguments."""

    num_blocks: int = 48
    num_heads: int = 16
    dropout: float = 0.0
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    use_trifast: bool = True


@dataclass
class MSAModuleArgs:
    """MSA module arguments."""

    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.0
    z_dropout: float = 0.0
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False
    use_trifast: bool = True


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@dataclass
class BoltzSteeringParams:
    """Steering parameters."""

    fk_steering: bool = True
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    guidance_update: bool = True
    num_gd_steps: int = 16


@rank_zero_only
def download(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.

    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(MODEL_URL, str(model))  # noqa: S310


def get_cache_path() -> str:
    """Determine the cache path, prioritising the BOLTZ_CACHE environment variable.

    Returns
    -------
    str: Path
        Path to use for boltz cache location.

    """
    env_cache = os.environ.get("BOLTZ_CACHE")
    if env_cache:
        resolved_cache = Path(env_cache).expanduser().resolve()
        if not resolved_cache.is_absolute():
            raise ValueError(f"BOLTZ_CACHE must be an absolute path, got: {env_cache}")
        return str(resolved_cache)

    return str(Path("~/.boltz").expanduser())


def check_inputs(
    data: Path,
    outdir: Path,
    override: bool = False,
) -> list[Path]:
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        filtered_data = []
        for d in data:
            if d.suffix in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                filtered_data.append(d)
            elif d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

        data = filtered_data
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()}

    # Remove them from the input data
    if existing and not override:
        data = [d for d in data if d.stem not in existing]
        num_skipped = len(existing) - len(data)
        msg = (
            f"Found some existing predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing predictions, will override."
        click.echo(msg)

    return data

def generate_neighbors(x, threshold=0.95, num_neighbors=4):
    """Courtesy: Willis Ma"""
    rng = np.random.Generator(np.random.PCG64())
    x_f = x.flatten(1)
    x_norm = torch.linalg.norm(x_f, dim=-1, keepdim=True, dtype=torch.float64).unsqueeze(-2)
    u = x_f.unsqueeze(-2) / x_norm.clamp_min(1e-12)
    v = torch.from_numpy(rng.standard_normal(size=(u.shape[0], num_neighbors, u.shape[-1]), dtype=np.float64)).to(
        u.device
    )
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
    random_noise = torch.randn(num_neighbors, B, flattened_dim, device=base_noise.device)

    # Project onto sphere defined by cosine similarity threshold
    base_norm = F.normalize(base_flat, dim=-1)  # [B, D]

    neighbors = []
    for i in range(num_neighbors):
        random_noise = torch.randn(B, flattened_dim, device=base_noise.device)

        proj = (random_noise * base_norm).sum(dim=-1, keepdim=True) * base_norm
        noise_orthogonal = random_noise - proj
        noise_orthogonal_normed = F.normalize(noise_orthogonal, dim=-1)

        neighbor_flat = threshold * base_norm + torch.sqrt(torch.tensor(1 - threshold**2, device=base_noise.device)) * noise_orthogonal_normed
        neighbor_flat_scaled = neighbor_flat * base_flat.norm(dim=-1, keepdim=True)

        neighbor = neighbor_flat_scaled.view(B, M, C)
        neighbors.append(neighbor)
    return torch.stack(neighbors)


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.

    """
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


@rank_zero_only
def process_inputs(  # noqa: C901, PLR0912, PLR0915
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 4096,
    use_msa_server: bool = False,
) -> None:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    click.echo("Processing input data.")
    existing_records = None

    # Check if manifest exists at output path
    manifest_path = out_dir / "processed" / "manifest.json"
    if manifest_path.exists():
        click.echo(f"Found a manifest file at output directory: {out_dir}")

        manifest: Manifest = Manifest.load(manifest_path)
        input_ids = [d.stem for d in data]
        existing_records = [
            record for record in manifest.records if record.id in input_ids
        ]
        processed_ids = [record.id for record in existing_records]

        # Check how many examples need to be processed
        missing = len(input_ids) - len(processed_ids)
        if not missing:
            click.echo("All examples in data are processed. Updating the manifest")
            # Dump updated manifest
            updated_manifest = Manifest(existing_records)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")
            return

        click.echo(f"{missing} missing ids. Preprocessing these ids")
        missing_ids = list(set(input_ids).difference(set(processed_ids)))
        data = [d for d in data if d.stem in missing_ids]
        assert len(data) == len(missing_ids)

    # Create output directories
    msa_dir = out_dir / "msa"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    processed_constraints_dir = out_dir / "processed" / "constraints"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    processed_constraints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    if existing_records is not None:
        click.echo(f"Found {len(existing_records)} records. Adding them to records")

    # Parse input data
    records: list[Record] = existing_records if existing_records is not None else []
    for path in tqdm(data):
        try:
            # Parse data
            if path.suffix in (".fa", ".fas", ".fasta"):
                target = parse_fasta(path, ccd)
            elif path.suffix in (".yml", ".yaml"):
                target = parse_yaml(path, ccd)
            elif path.is_dir():
                msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {path.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

            # Get target id
            target_id = target.record.id

            # Get all MSA ids and decide whether to generate MSA
            to_generate = {}
            prot_id = const.chain_type_ids["PROTEIN"]
            for chain in target.record.chains:
                # Add to generate list, assigning entity id
                if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                    entity_id = chain.entity_id
                    msa_id = f"{target_id}_{entity_id}"
                    to_generate[msa_id] = target.sequences[entity_id]
                    chain.msa_id = msa_dir / f"{msa_id}.csv"

                # We do not support msa generation for non-protein chains
                elif chain.msa_id == 0:
                    chain.msa_id = -1

            # Generate MSA
            if to_generate and not use_msa_server:
                msg = "Missing MSA's in input and --use_msa_server flag not set."
                raise RuntimeError(msg)

            if to_generate:
                msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
                click.echo(msg)
                compute_msa(
                    data=to_generate,
                    target_id=target_id,
                    msa_dir=msa_dir,
                    msa_server_url=msa_server_url,
                    msa_pairing_strategy=msa_pairing_strategy,
                )

            # Parse MSA data
            msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
            msa_id_map = {}
            for msa_idx, msa_id in enumerate(msas):
                # Check that raw MSA exists
                msa_path = Path(msa_id)
                if not msa_path.exists():
                    msg = f"MSA file {msa_path} not found."
                    raise FileNotFoundError(msg)

                # Dump processed MSA
                processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
                msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
                if not processed.exists():
                    # Parse A3M
                    if msa_path.suffix == ".a3m":
                        msa: MSA = parse_a3m(
                            msa_path,
                            taxonomy=None,
                            max_seqs=max_msa_seqs,
                        )
                    elif msa_path.suffix == ".csv":
                        msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                    else:
                        msg = f"MSA file {msa_path} not supported, only a3m or csv."
                        raise RuntimeError(msg)

                    msa.dump(processed)

            # Modify records to point to processed MSA
            for c in target.record.chains:
                if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                    c.msa_id = msa_id_map[c.msa_id]

            # Keep record
            records.append(target.record)

            # Dump structure
            struct_path = structure_dir / f"{target.record.id}.npz"
            target.structure.dump(struct_path)

            # Dump constraints
            constraints_path = processed_constraints_dir / f"{target.record.id}.npz"
            target.residue_constraints.dump(constraints_path)

        except Exception as e:
            if len(data) > 1:
                print(f"Failed to process {path}. Skipping. Error: {e}.")
            else:
                raise e

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")


@click.group()
def cli() -> None:
    """Boltz1."""
    return


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help="The directory where to download the data and model. Default is ~/.boltz, or $BOLTZ_CACHE if set.",
    default=get_cache_path,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--step_scale",
    type=float,
    help="The step size is related to the temperature at which the diffusion process samples the distribution."
    "The lower the higher the diversity among samples (recommended between 1 and 2). Default is 1.638.",
    default=1.638,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help="Pairing strategy to use. Used only if --use_msa_server is set. Options are 'greedy' and 'complete'",
    default="greedy",
)
@click.option(
    "--no_potentials",
    is_flag=True,
    help="Whether to not use potentials for steering. Default is False.",
)
def predict(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
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
) -> None:
    """Run predictions with Boltz-1."""
    global _boltz_predictions
    
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
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
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
        constraints_dir=(processed_dir / "constraints")
        if (processed_dir / "constraints").exists()
        else None,
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
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(pairformer_args),
        msa_module_args=asdict(msa_module_args),
        steering_args=asdict(steering_args),
    )
    model_module.eval()

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )

    # Compute predictions
    predictions = trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=True,
    )
    
    # Store predictions in global variable
    _boltz_predictions = predictions


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help="The directory where to download the data and model. Default is ~/.boltz, or $BOLTZ_CACHE if set.",
    default=get_cache_path,
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--step_scale",
    type=float,
    help="The step size is related to the temperature at which the diffusion process samples the distribution."
    "The lower the higher the diversity among samples (recommended between 1 and 2). Default is 1.638.",
    default=1.638,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help="Pairing strategy to use. Used only if --use_msa_server is set. Options are 'greedy' and 'complete'",
    default="greedy",
)
@click.option(
    "--no_potentials",
    is_flag=True,
    help="Whether to not use potentials for steering. Default is False.",
)
@click.option(
    "--num_candidates",
    type=int,
    default=8,
)



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
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
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
        constraints_dir=(processed_dir / "constraints")
        if (processed_dir / "constraints").exists()
        else None,
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
        "recycling_steps": 0,
        "sampling_steps": sampling_steps,
        "diffusion_samples": 1,
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

    num_iterations = 10
    perturbations_per_iter = 4
    threshold = 0.95

    for iteration in tqdm(range(num_iterations), desc="Zero-order optimization"):
        print(f"\n--- Starting Iteration {iteration} ---")
        top_candidate_noise = None
        top_score = -float("inf")

        neighbors = generate_protein_neighbors(
            base_noise, threshold=threshold, num_neighbors=perturbations_per_iter
        )

        previous_best_score = best_score  # Track the previous best score explicitly

        inner_iter = tqdm(range(perturbations_per_iter), desc=f"Iteration {iteration}", leave=False)
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
    print("\nFinal best PLDDT score:", best_score)
    writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format="mmcif",
    )
    writer.on_predict_batch_end(
        outputs=[best_out],
        batch=batch,
        batch_idx=0,
        dataloader_idx=0,
    )
    print("Best PLDDT score:", best_score)

@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="Path to save predictions.",
    default="./",
)
@click.option("--cache", default=get_cache_path, help="Cache directory.")
@click.option("--checkpoint", default=None, help="Optional checkpoint path.")
@click.option("--devices", default=1, help="Number of devices to use.")
@click.option("--accelerator", default="gpu", type=click.Choice(["gpu", "cpu", "tpu"]))
@click.option("--sampling_steps", default=200, help="Sampling steps.")
@click.option("--step_scale", default=1.638, help="Step size scale.")
@click.option("--num_workers", default=2, help="Dataloader workers.")
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--use_msa_server", is_flag=True, help="Use MMSeqs2 server for MSA generation.")
@click.option("--num_random_samples", default=10, help="Number of random samples.")
def random_sampling(
    data: str,
    out_dir: str,
    cache: str,
    checkpoint: Optional[str],
    devices: int,
    accelerator: str,
    sampling_steps: int,
    step_scale: float,
    num_workers: int,
    seed: Optional[int],
    use_msa_server: bool,
    num_random_samples: int,
) -> None:
    """Run random sampling predictions with Boltz-1."""
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")

    if seed is not None:
        seed_everything(seed)

    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser() / f"boltz_random_{data.stem}"
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
        constraints_dir=(processed_dir / "constraints")
        if (processed_dir / "constraints").exists()
        else None,
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
        "recycling_steps": 0,
        "sampling_steps": sampling_steps,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
    }
    diffusion_params = BoltzDiffusionParams(step_scale=step_scale)

    model_module: Boltz1 = Boltz1.load_from_checkpoint(
        checkpoint,
        strict=True,
        predict_args=predict_args,
        map_location="cuda" if accelerator == "gpu" else "cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        pairformer_args=asdict(PairformerArgs()),
        msa_module_args=asdict(MSAModuleArgs()),
        steering_args=asdict(BoltzSteeringParams()),
    )
    model_module.eval()

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
        score = out["ptm"].item()
        # score = out["plddt"].mean().item()
        random_scores.append(score)
        click.echo(f"Sample {i+1}: PLDDT = {score:.4f}")

        if score > best_score:
            best_score = score
            best_out = out

    click.echo("\nRandom Sampling Summary:")
    click.echo(f"Best PLDDT: {best_score:.4f}")
    click.echo(f"Worst PLDDT: {min(random_scores):.4f}")
    click.echo(f"Average PLDDT: {np.mean(random_scores):.4f}")

    writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format="mmcif",
    )
    writer.on_predict_batch_end([best_out], batch, 0, 0)

    click.echo(f"Best structure written. PLDDT: {best_score:.4f}")

if __name__ == "__main__":
    cli()