import torch
import pandas as pd
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import numpy as np

# Define dataset class
class ImageDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.dataframe = dataframe
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.dataframe.iloc[idx]['image_link'].split('/')[-1])
        if not os.path.isfile(img_name):
            print(f"Image not found: {img_name}. Skipping.")
            return None, None

        try:
            image = Image.open(img_name).convert('RGB')
        except OSError as e:
            print(f"Error loading image {img_name}: {e}. Skipping.")
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, ""

# Custom collate function to filter out None values
def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]  # Remove None entries
    if len(batch) == 0:
        return torch.zeros(0), []
    images, labels = zip(*batch)
    images = torch.stack(images)
    return images, labels

def extract_features(data_loader, model, device):
    features = []
    with torch.no_grad():
        for images, _ in tqdm(data_loader):
            if images.size(0) == 0:
                continue
            try:
                images = images.to(device)
                outputs = model(images)
                features.append(outputs.cpu().numpy())
            except Exception as e:
                print(f"Error during feature extraction: {e}")
                continue
    return np.vstack(features)

def main():
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")

    device = torch.device("cuda")

    # Paths to dataset files
    test_path = 'C:/Users/divya/OneDrive/Documents/student_resource 3/dataset/test.csv'  # Adjust to correct path
    image_folder = 'C:/Users/divya/OneDrive/Documents/student_resource 3/images'  # Folder where images are stored

    # Load testing dataset
    test_df = pd.read_csv(test_path)

    # Check dataset size
    print(f"Number of images in testing dataset: {len(test_df)}")

    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Initialize the pre-trained ResNet model
    model = models.resnet50(weights='DEFAULT')
    model = model.to(device)
    model.eval()

    # Create dataset and dataloader for testing
    batch_size = 64
    test_dataset = ImageDataset(test_df, image_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # Extract features for testing set
    print("Extracting features from testing set...")
    test_features = extract_features(test_loader, model, device)

    # Check feature dimension consistency
    feature_dim = test_features.shape[1]
    print(f"Feature dimension for test data: {feature_dim}")

    # Save testing features
    test_features_path = 'test_features.npy'
    np.save(test_features_path, test_features)
    print(f"Testing features saved to {test_features_path}.")

if __name__ == '__main__':
    main()
