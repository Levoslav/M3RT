from retriever import Retriever
import os
import torch
import torch.nn.functional as F
from PIL import Image
import pickle
import pandas as pd
import clip

class CLIPRetriever(Retriever):
    def __init__(self, IDs_in_integer_format=True) -> None:
        super().__init__()
        self.IDs_in_integer_format = IDs_in_integer_format
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

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
                if self.IDs_in_integer_format:
                    self.image_IDs.append(f'{int(image_name.strip("/").split("/")[-1].split(".")[0]):05d}') # Extract image id from path
                else:
                    self.image_IDs.append(image_name.strip("/").split("/")[-1].split(".")[0])

            preprocessed_images = []
            for image in images:
                image_input = self.preprocess(image).unsqueeze(0)
                preprocessed_images.append(image_input)
            preprocessed_images = torch.cat(preprocessed_images).to(self.device)
            del images

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
        text = clip.tokenize(text).to(self.device)

        # Encode text
        with torch.no_grad():
            encoded_text = F.normalize(self.model.encode_text(text), p=2, dim=-1)
        return encoded_text