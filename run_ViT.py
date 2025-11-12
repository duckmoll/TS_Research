
import subprocess
import sys
from run_parameters import Patch_ECL_S, Patch_ETTh1_S, Patch_traffic_S, Patch_ECL_L, Patch_ETTh1_L, Patch_traffic_L

default_dict = {
    "task_name": "long_term_forecast",
    "is_training": 1,
    "features": "M",
    "seq_len": 336,
    "label_len": 48,
    "des": "Exp",
    "train_epochs": 100,
    "itr": 1,
    "learning_rate": 1e-3,
    "lradj": "cosine",
}

folder_map = {
    "ETTh1": "ETT-small",
    "ETTh2": "ETT-small",
    "ETTm1": "ETT-small",
    "ETTm2": "ETT-small",
    "electricity": "electricity",
    "exchange_rate": "exchange_rate",
    "traffic": "traffic",
    "weather": "weather",
    "national_illness": "illness",
}

data_map = {
    "ETTh1": "ETTh1",
    "ETTh2": "ETTh2",
    "ETTm1": "ETTm1",
    "ETTm2": "ETTm2",
    "electricity": "custom",
    "exchange_rate": "custom",
    "traffic": "custom",
    "weather": "custom",
    "national_illness": "custom",
}

parameter_sets = []
for data in ["ETTh1", "electricity", "traffic"]:
    folder = folder_map[data]
    data_name = data_map[data]
    data_param_L = None
    data_param_S = None
    if data == "ETTh1":
        data_param_L = Patch_ETTh1_L
        data_param_S = Patch_ETTh1_S
    elif data == "electricity":
        data_param_L = Patch_ECL_L
        data_param_S = Patch_ECL_S
    elif data == "traffic":
        data_param_L = Patch_traffic_L
        data_param_S = Patch_traffic_S
    for pred_len in [96, 192, 336, 720]:
        new_params = [
            {
                **default_dict,
                **data_param_L,
                "root_path": f"../dataset/{folder}",
                "data_path": f"{data}.csv",
                "model_id": f"{data}_336_{pred_len}",
                "model": "ViT_Combined",
                "data": data_name,
                "pred_len": pred_len,
            },
        ]
        parameter_sets += new_params

python_executable = sys.executable

for params in parameter_sets:
    # Construct the command as a list of strings

    cmd = [
        python_executable,
        "/u/jliu61/TS_Research/run.py",
    ]

    for k, v in params.items():
        cmd.extend(["--" + str(k), str(v)])

    print(f"--- Running command: {' '.join(cmd)} ---")

    # Run the command
    # `check=True` will raise an error if the script fails
    print(cmd)
    subprocess.run(cmd, check=True)

print("All runs completed.")
