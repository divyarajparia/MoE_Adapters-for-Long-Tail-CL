import subprocess

imb_factors = [0.01]
log_paths = [
    "moe_cifar_imb0.01_intelfreeze.json",
]

for imb_factor, log_path in zip(imb_factors, log_paths):
    command = (
        "CUDA_VISIBLE_DEVICES=0 "
        "cd cil && "
        "python main.py "
        "--config-path configs/class "
        "--config-name cifar100_20-20-MoE-Adapters.yaml "
        "dataset_root=../datasets/ "
        "class_order=class_orders/cifar100.yaml "
        f"imb_factor={imb_factor} "
        f"log_path={log_path}"
    )
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

print("All runs executed.")