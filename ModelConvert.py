import torch
import timm

# створюємо ту ж саму архітектуру
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=11)
model.load_state_dict(torch.load("vit_tomato_model.pth", map_location="cpu"))
model.eval()

# тестовий вхід
example = torch.randn(1, 3, 224, 224)

# трасування (перетворення у TorchScript)
traced_model = torch.jit.trace(model, example)
traced_model.save("vit_tomato_model_lite.pt")
print("✅ Збережено у форматі TorchScript!")
