import subprocess
import os
import re
import socket
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Confirm the host machine explicitly
print(f"Running on host: {socket.gethostname()}")

# Explicitly set CUDA device to 4
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

fasta_folder = "/data/rbg/users/yxie25/FoldScaling/data/skempi/FASTA"
fasta_files = list(Path(fasta_folder).glob("*.fasta"))

boltz_cmd_template = "boltz random-sampling {} --use_msa_server"

def run_boltz(fasta_file):
    cmd = boltz_cmd_template.format(fasta_file)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        scores = [float(m.group(1)) for m in re.finditer(r"score:\s*([\d.]+)", result.stdout)]
        if scores:
            best, worst = max(scores), min(scores)
            diff = best - worst
            return fasta_file.name, best, worst, diff
        else:
            return fasta_file.name, None, None, None
    except subprocess.CalledProcessError as e:
        print(f"Error processing {fasta_file.name}: {e.stderr}")
        return fasta_file.name, None, None, None

if __name__ == "__main__":
    num_threads = 2

    results = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(run_boltz, fasta): fasta for fasta in fasta_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing FASTA files"):
            fasta_name, best, worst, diff = future.result()
            if best is not None:
                print(f"{fasta_name} | Best: {best:.3f}, Worst: {worst:.3f}, Diff: {diff:.3f}")
                results.append((fasta_name, best, worst, diff))
            else:
                print(f"No scores for {fasta_name}")

    # Save results
    import pandas as pd

    df = pd.DataFrame(results, columns=["FASTA", "Best", "Worst", "Diff"])
    df.to_csv("boltz_random_sampling_results.csv", index=False)
    print("Results saved to boltz_random_sampling_results.csv")
