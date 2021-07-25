import torch
from utils import resnet

model_dict = {
    "resnet18": resnet.resnet18,  # params: 11181642
    "resnet34": resnet.resnet34,  # params: 21289802
    "resnet50": resnet.resnet50,  # params: 23528522
    "resnet101": resnet.resnet101,  # params: 42520650
    "resnet152": resnet.resnet152,  # params: 58164298
}

def create_backbone(backbone, num_classes, use_pretrained):
    model_cls = model_dict[backbone]
    print(f"Using model {backbone}...", end='')

    model = model_cls(num_classes=num_classes, pretrained=use_pretrained)
    total_params = sum(p.numel() for p in model.parameters())
    layers = len(list(model.modules()))
    print(f" total parameters: {total_params}, layers {layers}")
    device_count = torch.cuda.device_count()

    if device_count > 0:
        model = model.to("cuda:0")
    else:
        model = model.to("cpu")
    return model
