import wandb
import torch
import matplotlib.pyplot as plt
import random
from src.network import SimpleClassifier
import src.config as cfg

print("WANDB_ENTITY:", cfg.WANDB_ENTITY)
print("WANDB_PROJECT:", cfg.WANDB_PROJECT)

def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def get_color(name):
    name = name.lower()
    if "efficientnet" in name:
        return "blue"
    elif "resnet" in name:
        return "red"
    else:
        return f"#{random.randint(0, 0xFFFFFF):06x}"

def main():
    api = wandb.Api()
    runs = api.runs(f"{cfg.WANDB_ENTITY}/{cfg.WANDB_PROJECT}")

    records_val, records_train = [], []

    for run in runs:
        model_name = run.config.get("model_name", run.config.get("MODEL_NAME"))
        acc_val = run.summary.get("accuracy/val") or run.summary.get("best_accuracy/val")
        acc_train = run.summary.get("accuracy/train") or run.summary.get("best_accuracy/train")
        if acc_val is None and acc_train is None:
            continue

        model = SimpleClassifier(
            model_name = model_name,
            num_classes = cfg.NUM_CLASSES,
            optimizer_params = cfg.OPTIMIZER_PARAMS,
            scheduler_params = cfg.SCHEDULER_PARAMS,
        )

        params_m = count_params(model)
        if acc_val is not None:
            records_val.append((model_name, params_m, acc_val))
        if acc_train is not None:
            records_train.append((model_name, params_m, acc_train))

    def plot_and_save(records, title, filename):
        if not records:
            print(f"{title}에 대해 유효한 실험 결과를 찾지 못했습니다.")
            return
        records.sort(key=lambda x: x[1])
        names, params, accs = zip(*records)
        colors = [get_color(name) for name in names]

        plt.figure(figsize=(6, 4))
        for name, p, a, c in zip(names, params, accs, colors):
            plt.scatter(p, a, color=c)
            plt.text(p, a, name, fontsize=8, va="bottom", ha="right", color=c)
        plt.xlabel("Parameters (Million)")
        plt.ylabel(f"{title} Accuracy")
        plt.title(f"Size–Accuracy Trade‑off ({title})")
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(filename, dpi=300)
        print(f"Saved trade‑off plot to {filename}")
        plt.close()

    plot_and_save(records_train, "Train", "size_accuracy_tradeoff_train.png")
    plot_and_save(records_val, "Validation", "size_accuracy_tradeoff_val.png")

if __name__ == "__main__":
    main()
