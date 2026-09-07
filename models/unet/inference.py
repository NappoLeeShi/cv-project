import torch
from PIL import Image
from torchvision import transforms


def load_model(model, weight_path=None, device="cpu"):

    model = model.to(device)

    if weight_path is not None:
        checkpoint = torch.load(
            weight_path,
            map_location=device
        )

        model.load_state_dict(checkpoint)

    model.eval()

    return model


def predict(model, image_path, device="cpu"):

    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    prediction = torch.argmax(
        output,
        dim=1
    )

    return prediction.squeeze(0).cpu()