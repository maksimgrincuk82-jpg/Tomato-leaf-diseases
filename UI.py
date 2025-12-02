import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button
import numpy as np

# --- завантаження моделі ---
MODEL_PATH = r"C:\Users\zalut\PycharmProjects\TomatoGPU_ViT\vit_tomato_model.pth"

# якщо ти використовувала timm:
import timm
model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=11)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# --- список класів (можна змінити під свій датасет) ---
labels = [
    "Bacterial_spot",
    "Early_blight",
    "healthy",
    "Late_blight",
    "Leaf_Mold",
    "powdery_mildew",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus"
]

# --- передобробка ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    return labels[pred.item()], conf.item()




# --- Tkinter UI ---
class App:
    def __init__(self, root):
        self.root = root
        root.title("Tomato Disease Classifier 🍅")
        root.geometry("500x600")
        root.configure(bg="#f3f4f6")

        self.label = Label(root, text="Оберіть зображення листка", font=("Arial", 14), bg="#f3f4f6")
        self.label.pack(pady=10)

        self.img_label = Label(root, bg="#f3f4f6")
        self.img_label.pack(pady=10)

        self.result_label = Label(root, text="", font=("Arial", 16, "bold"), bg="#f3f4f6")
        self.result_label.pack(pady=10)

        self.button = Button(root, text="📸 Вибрати фото", command=self.load_image,
                             font=("Arial", 12), bg="#4caf50", fg="white", width=20)
        self.button.pack(pady=20)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return

        # показуємо фото
        img = Image.open(file_path).resize((300, 300))
        photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=photo)
        self.img_label.image = photo

        # передбачення
        label, conf = predict(file_path)
        self.result_label.config(
            text=f"Результат: {label}\nЙмовірність: {conf:.2f}"
        )

# --- запуск ---
root = tk.Tk()
app = App(root)
root.mainloop()

