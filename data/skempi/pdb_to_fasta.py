import pathlib
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqIO import write
from Bio.SeqRecord import SeqRecord
import string

def pdb_to_boltz_fasta(pdb_file, fasta_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    ppb = PPBuilder()

    records = []
    chain_labels = iter(string.ascii_uppercase)  # A, B, C, D, ...
    
    for model in structure:
        for chain in model:
            polypeptides = ppb.build_peptides(chain)
            
            # Skip empty chains
            if not polypeptides:
                continue

            label = next(chain_labels, None)
            if label is None:
                raise ValueError("Exceeded maximum chain labels (26 chains)")

            # Concatenate all fragments within this chain into a single sequence
            full_seq = ''.join([str(poly.get_sequence()) for poly in polypeptides])

            record_id = f"{label}|protein|empty"
            records.append(SeqRecord(seq=full_seq, id=record_id, description=""))

    if records:
        write(records, fasta_file, "fasta")
    else:
        print(f"No sequences extracted from {pdb_file.name}")

pdbs_path = pathlib.Path("PDBs")
fasta_files_path = pathlib.Path("FASTA")
fasta_files_path.mkdir(parents=True, exist_ok=True)

for pdb_file in pdbs_path.glob("*.pdb"):
    fasta_file = fasta_files_path / (pdb_file.stem + ".fasta")
    print(f"Converting {pdb_file} to {fasta_file}")
    pdb_to_boltz_fasta(pdb_file, fasta_file)