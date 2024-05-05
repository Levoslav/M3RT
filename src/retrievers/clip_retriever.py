from retriever import Retriever
import os
import torch
import torch.nn.functional as F
from PIL import Image
import pickle
import pandas as pd
import clip

class CLIPRetriever(Retriever):
    def __init__(self) -> None:
        super().__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def encode_images(self, images_paths ,out_dir=None, batch_size=50):
        batches  = [images_paths[i:i+batch_size] for i in range(0, len(images_paths), batch_size)]
        self.image_IDs = []
        self.image_encodings = None
        for batch in batches:
            images = []

            # Preprocess Images
            for image_name in batch:
                images.append(Image.open(image_name))
                self.image_IDs.append(image_name)

            preprocessed_images = []
            for image in images:
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                preprocessed_images.append(image_input)
            preprocessed_images = torch.cat(preprocessed_images)
            del images

            # Encode Images
            with torch.no_grad():
                if self.image_encodings is None:
                    self.image_encodings = self.model.encode_image(preprocessed_images)
                else:
                    self.image_encodings = torch.cat((self.image_encodings, self.model.encode_image(preprocessed_images)), dim=0)

        self.image_encodings = F.normalize(self.image_encodings, p=2, dim=-1)
        # Save if out_dir specified
        if out_dir is not None and self.image_encodings is not None:
            torch.save(self.image_encodings, out_dir + '/CLIP_image_encodings.pth')
            with open(out_dir + '/CLIP_images_IDs.pkl', 'wb') as f:
                pickle.dump(self.image_IDs, f)

    def encode_text(self, text):
        # Preprocess text
        text = clip.tokenize(text).to(self.device)

        # Encode text
        with torch.no_grad():
            encoded_text = F.normalize(self.model.encode_text(text), p=2, dim=-1)
        return encoded_text

    def load_encoded_images(self, directory):
        # Load tensor from file
        self.image_encodings = torch.load(os.path.join(directory ,'CLIP_image_encodings.pth'))
        # Load list from file
        with open(os.path.join(directory , 'CLIP_images_IDs.pkl'), 'rb') as f:
            self.image_IDs = pickle.load(f)

    def compute_cumulation(self, labels: pd.DataFrame, out_dir=None):
        ranks_cumulation = [0] * len(self.image_IDs)
        for id, label in zip(labels.ID, labels.label):
            ordered_images = self.compare_to_images(label)
            rank = self._find_rank(ordered_images, id)
            ranks_cumulation[rank:] = [x+1 for x in ranks_cumulation[rank:]]
            # ranks_cumulation[rank] += 1

            # Save if out_dir specified
            if out_dir is not None :
                dir = os.path.join(out_dir , 'CLIP_cumulation.pkl')
                with open(dir , 'wb') as f:
                    pickle.dump(ranks_cumulation, f)
        return ranks_cumulation