# test_model_names_fixed.py
import timm

model_names = [
    # ResNet 계열
    "resnet34", "resnet50", "resnet152",

    # Inception 계열 (inception_v2 → inception_v3로 대체)
    "inception_v3",
    "inception_resnet_v2",

    # Xception
    "xception",

    # DenseNet
    "densenet201",

    # ResNeXt
    "resnext101_32x8d",

    # SENet
    "senet154",

    # NASNet-A
    "nasnetalarge",

    # AmoebaNet
    # timm에서는 접미사 번호가 붙은 버전만 지원하므로,
    # A∞ → amoebanet_a0, C∞ → amoebanet_c0 등 구체 버전을 사용
    "amoebanet_a0",
    "amoebanet_c0",

    # EfficientNet B0~B7
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    "efficientnet_b3", "efficientnet_b4", "efficientnet_b5",
    "efficientnet_b6", "efficientnet_b7",
]

if __name__ == "__main__":
    for name in model_names:
        try:
            timm.create_model(name, pretrained=False)
            print(f"[ OK ] {name}")
        except Exception as e:
            print(f"[FAIL] {name} -> {e}")
