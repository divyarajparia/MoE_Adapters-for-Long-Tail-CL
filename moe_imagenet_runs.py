import subprocess

imb_factors = [0.01, 0.02, 0.1]
log_paths = [
    "moe_imagenet_final_imb0.01.json",
    "moe_imagenet_final_imb0.05.json",
    "moe_imagenet_final_imb0.1.json"
]

for imb_factor, log_path in zip(imb_factors, log_paths):
    command = (
        "CUDA_VISIBLE_DEVICES=1 "
        "cd cil && "
        "python main.py "
        "--config-path configs/class "
        "--config-name imagenet100_20-20.yaml "
        "dataset_root=../datasets/ "
        "class_order=class_orders/imagenet100.yaml "
        f"imb_factor={imb_factor} "
        f"log_path={log_path}"
    )
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

print("All runs executed.")