# script for cleaning up monomer files (one-time use only)
from pathlib import Path
import shutil

def process_fasta_and_a3m(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_files = list(input_dir.glob("*.fasta"))

    for fasta_path in fasta_files:
        fasta_name = fasta_path.stem  # e.g., "1a32"
        a3m_path = input_dir / f"{fasta_name}.a3m"

        # Read fasta content
        with open(fasta_path, "r") as f:
            lines = f.read().strip().splitlines()
            if not lines or len(lines) < 2:
                print(f"Skipping {fasta_path.name}: invalid or empty FASTA format.")
                continue

            header = lines[0]
            sequence = lines[1]

        # Parse ID
        id_part = header.split("|")[0][1:]  # remove '>' from beginning

        # Generate new headers
        new_header_with_msa = f">{id_part}|protein|{id_part}.a3m"
        new_header_no_msa = f">{id_part}|protein|empty"

        # Write with MSA version
        with open(output_dir / f"{id_part}_with_msa.fasta", "w") as f:
            f.write(f"{new_header_with_msa}\n{sequence}\n")

        # Write no-MSA version (with "empty")
        with open(output_dir / f"{id_part}_no_msa.fasta", "w") as f:
            f.write(f"{new_header_no_msa}\n{sequence}\n")

        # Copy over the .a3m file as-is (if it exists)
        if a3m_path.exists():
            shutil.copy(a3m_path, output_dir / f"{id_part}.a3m")
        else:
            print(f"Warning: .a3m file not found for {id_part}")

    print(f"\nCleaned FASTA files and .a3m files saved to: {output_dir}")


process_fasta_and_a3m("monomers", "monomers_cleaned")