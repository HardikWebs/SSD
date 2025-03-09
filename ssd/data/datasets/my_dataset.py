import os
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import KittiFlow
from ssd.structures.container import Container

class MyDataset(torch.utils.data.Dataset):
    class_names = ('__background__', 'Car', 'Van', 'Truck', 'Pedestrian', 'Cyclist')

    def __init__(self, root, split='training', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Load image paths and annotation paths
        self.image_dir = os.path.join(root, split, 'image_2')
        self.label_dir = os.path.join(root, split, 'label_2')
        self.image_files = sorted(os.listdir(self.image_dir))
        self.label_files = sorted(os.listdir(self.label_dir))
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image
        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Load annotations (bounding boxes and labels)
        label_path = os.path.join(self.label_dir, self.label_files[index])
        boxes, labels = self._parse_kitti_annotations(label_path)

        # Apply transforms
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        # Pack targets into a Container
        targets = Container(
            boxes=boxes,
            labels=labels,
        )

        # Return image, targets, and index
        return image, targets, index
    
    def get_image_dimensions(self, index):
        """
        Returns the width and height of the image at the specified index.
        Args:
            index (int): Index of the image in the dataset.
        Returns:
            tuple: (width, height) of the image.
        """
        image_path = os.path.join(self.image_dir, self.image_files[index])
        with Image.open(image_path) as img:
            width, height = img.size
        return width, height

    def _parse_kitti_annotations(self, label_path):
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] in self.class_names:  # Filter relevant classes
                    # KITTI format: [class, truncated, occluded, alpha, bbox_x1, bbox_y1, bbox_x2, bbox_y2, ...]
                    x1, y1, x2, y2 = map(float, parts[4:8])
                    boxes.append([x1, y1, x2, y2])
                    labels.append(self.class_dict[parts[0]])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        return boxes, labels

    # def _class_name_to_id(self, class_name):
    #     # Map class names to integer labels
    #     class_to_id = {"Car": 0, "Van": 1, "Truck": 2, "Pedestrian": 3, "Cyclist": 4}
    #     return class_to_id.get(class_name, -1)

# Example usage
# if __name__ == "__main__":
#     # Path to the KITTI dataset
#     kitti_root = '/path/to/kitti/dataset'

#     # Initialize the dataset
#     dataset = KITTIDataset(root=kitti_root, split='training')

#     # Access a sample
#     image, targets, index = dataset[0]
#     print("Image shape:", image.size)
#     print("Boxes:", targets.boxes)
#     print("Labels:", targets.labels)