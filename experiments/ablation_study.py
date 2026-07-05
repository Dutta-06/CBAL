# A simple wrapper that runs run_cbal_full.py with different configs
import os

configs = [
    "configs/base_config.yaml",
    # You would create these variants:
    # "configs/ablation_no_acoustic.yaml", 
    # "configs/ablation_no_context.yaml"
]

meeting_id = "IS1009a"

for conf in configs:
    print(f"Running ablation: {conf}")
    os.system(f"python -m experiments.run_cbal_full --meeting_id {meeting_id} --config {conf}")