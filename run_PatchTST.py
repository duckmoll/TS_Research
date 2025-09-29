import subprocess
import sys

default_dict = {
    "task_name": "long_term_forecast",
    "is_training": 1,
    "features": "S",
    "seq_len": 336,
    "label_len": 48,
    "e_layers": 3,
    "d_layers": 1,
    "factor": 1,
    "enc_in": 1,
    "dec_in": 7,
    "c_out": 1,
    "des": "Exp",
    "n_heads": 4,
    "d_ff": 128,
    "d_model": 16,
    "train_epochs": 100,
    "batch_size": 128,
    "dropout": 0.3,
    "itr": 1,
    "patch_len": 16,
    "patch_stride": 8,
    "learning_rate": 1e-4,
    "lradj": "type3",
    "moving_avg_type": "moe"
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
    for pred_len in [96, 192, 336, 720]:
        for k in [1, 3, 5, 7, 9]:
            new_params = [
                {
                    **default_dict,
                    "root_path": f"../dataset/{folder}",
                    "data_path": f"{data}.csv",
                    "model_id": f"{data}_336_{pred_len}_moe_patch",
                    "model": "PatchTST_MoE",
                    "data": data_name,
                    "pred_len": pred_len,
                    "moe_topk": k,
                },
            ]
        parameter_sets += new_params

python_executable = sys.executable

for params in parameter_sets:
    # Construct the command as a list of strings

    cmd = [
        python_executable,
        "D:\\Projects\\TS_Research\\run.py",
    ]

    for k, v in params.items():
        cmd.extend(["--" + str(k), str(v)])

    print(f"--- Running command: {' '.join(cmd)} ---")

    # Run the command
    # `check=True` will raise an error if the script fails
    subprocess.run(cmd, check=True)

print("All runs completed.")
