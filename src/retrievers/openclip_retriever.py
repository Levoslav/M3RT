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

        print(f"moving model to device: {self.device}")
        self.model = self.model.to(self.device)
        print("Inicialization finished")
        
    def encode_images(self, images_paths ,out_file_path=None, batch_size=500):
        batches  = [images_paths[i:i+batch_size] for i in range(0, len(images_paths), batch_size)]
        self.image_IDs = []
        self.image_encodings = None
        self._print_runtime_message(message_type='images_encoding_start', batch_size=batch_size, num_of_batches=len(batches))
        for i, batch in enumerate(batches):
            images = []

            # Preprocess Images
            for image_name in batch:
                images.append(Image.open(image_name))
                self.image_IDs.append(image_name)

            preprocessed_images = []
            for image in images:
                image_input = self.preprocess(image).unsqueeze(0)
                preprocessed_images.append(image_input)
            preprocessed_images = torch.cat(preprocessed_images).to(self.device)
            del images
            print(f"   batch {i+1}. preprocessed")

            # Encode Images
            with torch.no_grad():
                if self.image_encodings is None:
                    self.image_encodings = self.model.encode_image(preprocessed_images)
                else:
                    self.image_encodings = torch.cat((self.image_encodings, self.model.encode_image(preprocessed_images)), dim=0)

            self._print_runtime_message(message_type='batch_encoded',batch_num=i+1)

        self.image_encodings = F.normalize(self.image_encodings, p=2, dim=-1)
        # Save if out_file_path specified
        if out_file_path is not None and self.image_encodings is not None:
            self.image_encodings = self.image_encodings.cpu() # Move to cpu to save
            with open(out_file_path, 'wb') as f:
                pickle.dump((self.image_IDs, self.image_encodings), f)

    def encode_text(self, text):
        # Preprocess text
        text = self.tokenizer(text).to(self.device)

        # Encode text
        with torch.no_grad():
            encoded_text = F.normalize(self.model.encode_text(text), p=2, dim=-1)
        return encoded_text