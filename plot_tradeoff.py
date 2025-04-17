# size-accuracy trade-off plot 용도

import wandb
import torch
import matplotlib.pyplot as plt
from src.network import SimpleClassifier
import src.config as cfg
print("▶ WANDB_ENTITY:", cfg.WANDB_ENTITY)
print("▶ WANDB_PROJECT:", cfg.WANDB_PROJECT)

def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def main():
    # 1) W&B API 초기화
    api = wandb.Api()

    # 2) 지정한 entity/project 의 모든 run 불러오기
    runs = api.runs(f"{cfg.WANDB_ENTITY}/{cfg.WANDB_PROJECT}")

    records = []
    for run in runs:
        # 3) 각 run 의 config 에서 모델 이름, summary 에서 최종 val accuracy 가져오기
        model_name = run.config.get("model_name", run.config.get("MODEL_NAME"))
        acc = run.summary.get("accuracy/val") or run.summary.get("best_accuracy/val")
        if acc is None:
            continue

        # 4) 해당 모델을 SimpleClassifier 래퍼로 인스턴스화
        model = SimpleClassifier(
            model_name = model_name,
            num_classes = cfg.NUM_CLASSES,
            optimizer_params = cfg.OPTIMIZER_PARAMS,
            scheduler_params = cfg.SCHEDULER_PARAMS,
        )

        # 5) 파라미터 수 계산
        params_m = count_params(model)
        records.append((model_name, params_m, acc))

    if not records:
        print("▶️ 유효한 실험 결과를 찾지 못했습니다.")
        return

    # 6) 파라미터 수 기준 오름차순 정렬
    records.sort(key=lambda x: x[1])
    names, params, accs = zip(*records)

    # 7) 플롯 그리기
    plt.figure(figsize=(6,4))
    plt.scatter(params, accs)
    for n, p, a in zip(names, params, accs):
        plt.text(p, a, n, fontsize=8, va="bottom", ha="right")
    plt.xlabel("Parameters (Million)")
    plt.ylabel("Validation Accuracy")
    plt.title("Size–Accuracy Trade‑off")
    plt.grid(True)
    plt.tight_layout()

    # PNG로 저장
    output_path = "size_accuracy_tradeoff.png"
    plt.savefig(output_path, dpi=300)
    print(f"▶ Saved trade‑off plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
