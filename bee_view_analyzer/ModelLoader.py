import os.path

from torchvision.datasets import ImageFolder
from torchvision import transforms

import torch
from torch.utils.data import DataLoader

class ModelLoader:
    
    def __init__(self, base_model, base_model_path, bee_view_image_path, batch_size, image_height, image_width):
        self.batch_size = batch_size
        
        trans = transforms.Compose(
            [transforms.Resize((image_height, image_width), interpolation=5), 
             transforms.ToTensor()]
        )

        image_folder = ImageFolder(bee_view_image_path, transform=trans)
        self.data_loader = DataLoader(image_folder, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if os.path.isfile(base_model_path):
            self.model = base_model.load_model(vae_num_latents=64, vae_num_hidden=128, path=base_model_path)
        else:
            self.model = base_model.train_model()
            base_model.save_model(self.model, base_model_path)
        
        self.model = self.model.to(self.device)
        
    def get_image_count(self):
        return len(self.data_loader) * self.batch_size
