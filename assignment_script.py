import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# Custom dataset for baseball detection
# Uses a single video frame and a target box for the baseball in the frame.
class BaseballFrameDataset(Dataset):
    def __init__(self, videos_dir, annotations_dir, image_size=112, transform=None):
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.image_size = image_size
        self.transform = transform

        self.samples = []
        self._build_index()

    # Helps parse all XML file in annotations_fir and creates a list of samples.
    def _build_index(self):
        xml_files = [
            f for f in os.listdir(self.annotations_dir)
            if f.lower().endswith(".xml")
        ]
        xml_files.sort()

        # Finds matching video files for a given base name.
        def find_video_path(base_name):
            for ext in (".mov", ".mp4", ".avi"):
                candidate = os.path.join(self.videos_dir, base_name + ext)
                if os.path.exists(candidate):
                    return candidate
            return None
        
        for xml_name in xml_files:
            xml_path = os.path.join(self.annotations_dir, xml_name)
            base_name = os.path.splitext(xml_name)[0]

            video_path = find_video_path(base_name)
            if video_path is None:
                continue # skips if there is no matching video.

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            except Exception as e:
                print(f"Warning: Wasn't able to parse {xml_path}: {e}")
                continue

            boxes_per_frame = {}

            #CVAT makes use of <tracks> with <box> elements.
            for track in root.findall("track"):
                label = track.get("label", "")
                if label != "baseball":
                    continue
                for box in track.findall("box"):
                    # outside=0: Box is active i.e. ball is present
                    if box.get("outside", "0") != "0":
                        continue
                    frame_idx = int(box.get("frame"))
                    xtl = float(box.get("xtl"))
                    ytl = float(box.get("ytl"))
                    xbr = float(box.get("xbr"))
                    ybr = float(box.get("ybr"))
                    boxes_per_frame.setdefault(frame_idx, []).append((xtl, ytl, xbr, ybr))

            # A sample is created for every frame that has at least one baseball box.
            # Only the first box is used in case of multiple boxes (usually only has one ball).
            for frame_idx in sorted(boxes_per_frame.keys()):
                bbox = boxes_per_frame[frame_idx][0]
                self.samples.append(
                    {
                        "video_path": video_path,
                        "frame_idx": frame_idx,
                        "bbox": bbox, # original video resolution
                    }
                )
        print(f"Dataset built with {len(self.samples)} labelled frames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        frame_idx = sample["frame_idx"]
        xtl, ytl, xbr, ybr = sample["bbox"]

        # Reading the frame from the video.
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        # In case the frame can't be read, a blank image is used.
        if not ret or frame is None:
            frame = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            frame_h, frame_w = self.image_size, self.image_size
        else:
            # If the frame is BGR, convert to RBG.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_h, frame_w = frame.shape[:2]
            frame = cv2.resize(frame, (self.image_size, self.image_size))

        # Converting the image to tensor [3, H, W] normalized to [0, 1].
        img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        # Converting the bounding box to relative coordinates (0-1) with respect to the original frame size.
        xtl_rel = xtl/ frame_w
        xbr_rel = xbr / frame_w
        ytl_rel = ytl / frame_h
        ybr_rel = ybr / frame_h

        target = torch.tensor([xtl_rel, ytl_rel, xbr_rel, ybr_rel], dtype=torch.float32)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

# Simple CNN Regressor model for bounding boxes.
# Takes an image and predicts 4 numbers in [0, 1].
class BaseballNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((7, 7))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(), # keeps the output in the [0,1] range.
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.regressor(x)
        return x

# Training and evaluation.
def train_model(model, train_loader, val_loader=None, epochs=5, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] - Train SmoothL1 Loss: {avg_train_loss:.4f}")

        # Saving the model
        torch.save(model.state_dict(), "baseball_model.pth")
        print("The model has been saved as baseball_model.pth")

def evaluate_model(model, data_loader, device=None, silent=False):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.SmoothL1Loss(reduction="sum")
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader.dataset)

    if not silent:
        print(f"Average SmoothL1 loss on dataset: {avg_loss:.4f}")

    return avg_loss

# Main
if __name__=="__main__":
    VIDEOS_DIR = "Raw Data"
    ANN_DIR = "Annotations"

    # Creating a dataset from all the annotated frames.
    dataset = BaseballFrameDataset(
        videos_dir=VIDEOS_DIR,
        annotations_dir=ANN_DIR,
        image_size=112,
        transform=None,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "The dataset is empty. Check that the folders exist and that the filenames match."
        )

    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                        generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = BaseballNet()
    train_model(model, train_loader, val_loader=val_loader, epochs=5, lr=1e-3)

    print("\nFinal evaluation: ")
    evaluate_model(model, val_loader)

# Evaluation.
if __name__ == "__main__":
    VIDEOS_DIR = "Raw Data"
    ANN_DIR = "Annotations"

    # Rebuilding the dataset exactly as in training.
    dataset = BaseballFrameDataset(
        videos_dir=VIDEOS_DIR,
        annotations_dir=ANN_DIR,
        image_size=112,
        transform=None,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "The dataset is empty. Check that the folders existand that you created baseball_model.pth."
        )

    # Using the same 80/20 split with the same seed.
    generator = torch.Generator().manual_seed(42)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size],
                                  generator=generator)

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading the model
    model = BaseballNet()
    model.load_state_dict(torch.load("baseball_model.pth", map_location=device))

    print("Loaded baseball_model.pth, evaluating on validation split.")
    avg_loss = evaluate_model(model, val_loader, device=device)
    print(f"Average SmoothL1 loss on validation set: {avg_loss:.4f}")




