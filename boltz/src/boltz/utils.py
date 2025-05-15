from Bio.PDB.MMCIFParser import MMCIFParser
import click
import numpy as np
import torch
from scipy.spatial.distance import cdist
import requests
import pathlib


def extract_coords_from_cif(file_path):
    parser = MMCIFParser()
    structure = parser.get_structure("protein", file_path)

    residues = [res for res in structure.get_residues() if "CA" in res]
    coords = np.array([res["CA"].get_coord() for res in residues])
    residue_ids = [res.get_id()[1] for res in residues]

    return coords, residue_ids


def align_coords(coords_pred, ids_pred, coords_true, ids_true):
    common_ids = sorted(set(ids_pred) & set(ids_true))
    idx_pred = [ids_pred.index(i) for i in common_ids]
    idx_true = [ids_true.index(i) for i in common_ids]
    return coords_pred[idx_pred], coords_true[idx_true]


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


def compute_lddt(cif_pred, cif_true, cutoff=15.0, per_atom=False):
    # Extract coordinates and residue IDs
    coords_pred, ids_pred = extract_coords_from_cif(cif_pred)
    coords_true, ids_true = extract_coords_from_cif(cif_true)

    # Align coordinates by residue IDs
    coords_pred_aligned, coords_true_aligned = align_coords(
        coords_pred, ids_pred, coords_true, ids_true
    )

    # Convert to distance matrices
    dmat_pred = torch.tensor(cdist(coords_pred_aligned, coords_pred_aligned))
    dmat_true = torch.tensor(cdist(coords_true_aligned, coords_true_aligned))

    # Compute mask (excluding self-distances)
    n_atoms = dmat_true.shape[0]
    mask = 1 - torch.eye(n_atoms)

    return lddt_dist(dmat_pred, dmat_true, mask, cutoff, per_atom)


@click.group()
def cli() -> None:
    """Utils for Boltz-1."""
    return


@cli.command()
@click.argument("pdb_id_file", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path(exists=False))
def dld_cif(pdb_id_file, out_dir):
    """
    Download .cif files for a list of PDB IDs specified in a text file.
    Skips downloading if the file already exists.

    Args:
        pdb_id_file (str | Path): Path to a text file where each line contains a PDB ID.
        out_dir (str | Path): Path to the output directory where .cif files will be saved.
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://files.rcsb.org/download"

    # Read PDB IDs from file, skipping empty lines and stripping whitespace
    with open(pdb_id_file, "r") as f:
        pdb_ids = [line.strip().upper() for line in f if line.strip()]

    for pdb_id in pdb_ids:
        pdb_id = pdb_id.lower()
        output_path = out_dir / f"{pdb_id}.cif"

        if output_path.exists():
            print(f"Skipping {pdb_id}: file already exists at {output_path}")
            continue

        url = f"{base_url}/{pdb_id}.cif"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded {pdb_id}.cif to {output_path}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {pdb_id}: {e}")


if __name__ == "__main__":
    cli()
