from retriever import Retriever
import os
import torch
import torch.nn.functional as F
from PIL import Image
import pickle
import pandas as pd
import open_clip
import sys


class OpenCLIPRetriever(Retriever):
    def __init__(self, version) -> None:
        super().__init__()
        self.version = version
        if self.version == 'ViT_G_14_laion2b':
            self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s34b_b88k')
            self.tokenizer = open_clip.get_tokenizer('ViT-g-14')
        elif self.version == 'ViT_SO400M_14_webli':
            self.model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
            self.tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
        elif self.version == 'ViT_H_14_dfn5b':
            self.model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')
            self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        elif self.version == 'ViT_L_16_webli':
            self.model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-L-16-SigLIP-384')
            self.tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-L-16-SigLIP-384')
        elif self.version == 'ViT_B_16_webli':
            self.model, self.preprocess = open_clip.create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP-512')
            self.tokenizer = open_clip.get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP-512')
        elif self.version == 'ViT_H_14_laion2b':
            self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
            self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        else:
            print(f'Unknown open_clip version: "{self.version}"')
            sys.exit(1)
        

    def encode_images(self, images_paths ,out_dir=None, batch_size=500):
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
            torch.save(self.image_encodings, out_dir + f'/OpenCLIP_image_encodings_{self.version}.pth')
            with open(out_dir + f'/OpenCLIP_images_IDs_{self.version}.pkl', 'wb') as f:
                pickle.dump(self.image_IDs, f)

    def encode_text(self, text):
        # Preprocess text
        text = self.tokenizer(text).to(self.device)

        # Encode text
        with torch.no_grad():
            encoded_text = F.normalize(self.model.encode_text(text), p=2, dim=-1)
        return encoded_text

    def load_encoded_images(self, directory):
        # Load tensor from file
        self.image_encodings = torch.load(os.path.join(directory ,f'OpenCLIP_image_encodings_{self.version}.pth'))
        # Load list from file
        with open(os.path.join(directory , f'OpenCLIP_images_IDs_{self.version}.pkl'), 'rb') as f:
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
                dir = os.path.join(out_dir , f'OpenCLIP_cumulation_{self.version}.pkl')
                with open(dir , 'wb') as f:
                    pickle.dump(ranks_cumulation, f)
        return ranks_cumulation