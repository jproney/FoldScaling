import os
import glob
import subprocess
import csv
import re
import tempfile
from Bio.PDB import PDBParser, PDBIO, Select

# TMscore executable and directories
tm_exec = "/home/gridsan/jroney/solab/af3/FoldScaling/boltz/tts_scripts/TMscore"
native_dir =  "/home/gridsan/jroney/solab/af3/FoldScaling/data/monomers_pdb"

sources = {
    "fk": "/home/gridsan/jroney/solab/af3/FoldScaling/data/monomer_predictions_fk/boltz_results_monomers_fasta/predictions",
    "unguided": "/home/gridsan/jroney/solab/af3/FoldScaling/data/monomer_predictions_unguided/boltz_results_monomers_fasta/predictions"
}

# Only allow codes found in FK
allowed_codes = set(os.listdir(sources["fk"]))

# Renumbering utility
class ResidueRenumberSelect(Select):
    def __init__(self):
        self.new_id = 1
    def accept_residue(self, residue):
        residue.id = (" ", self.new_id, " ")
        self.new_id += 1
        return True

def renumber_pdb(input_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("native", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb", mode="w")
    io.save(tmp.name, ResidueRenumberSelect())
    return tmp.name

# Main loop
results = []
for label, base_dir in sources.items():
    cif_pattern = os.path.join(base_dir, "*", "*_model_*.cif")
    for cif_path in glob.glob(cif_pattern):
        filename = os.path.basename(cif_path)
        parent = os.path.basename(os.path.dirname(cif_path))

        if parent not in allowed_codes:
            continue  # skip codes not in FK

        match = re.search(r"_model_(\d+).cif", filename)
        if not match:
            continue
        model_num = match.group(1)
        code = parent
        native_path = os.path.join(native_dir, f"{code}.pdb")

        if not os.path.isfile(native_path):
            print(f"Missing native: {code}")
            continue

        renum_pdb = renumber_pdb(native_path)

        try:
            result = subprocess.run(
                [tm_exec, cif_path, renum_pdb],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True
            )
            output = result.stdout
            tm_match = re.search(r"TM-score\s*=\s*([\d.]+)", output)
            if tm_match:
                tm_score = float(tm_match.group(1))
                results.append((code, model_num, label, tm_score))
                print(results[-1])
            else:
                print(f"No TM-score found for {cif_path}")
        except subprocess.CalledProcessError:
            print(f"TMscore failed for {cif_path}")
        finally:
            os.unlink(renum_pdb)

# Write results to CSV
with open("tm_scores.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["code", "model_num", "source", "tm_score"])
    writer.writerows(results)

print("Done. Results saved to tm_scores.csv.")

