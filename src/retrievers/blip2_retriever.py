from retriever import Retriever
import torch
import os
import torch.nn.functional as F
from PIL import Image
import pickle
import pandas as pd
from lavis.models import load_model_and_preprocess


class BLIP2Retriever(Retriever):
    def __init__(self) -> None:
        super().__init__()
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=self.device)

    def encode_images(self, images_paths, out_file_path=None, batch_size=500):
        batches  = [images_paths[i:i+batch_size] for i in range(0, len(images_paths), batch_size)]
        self.image_IDs = []
        self.image_encodings = None
        self._print_runtime_message(message_type='images_encoding_start', batch_size=batch_size, num_of_batches=len(batches))
        for i, batch in enumerate(batches):
            images = []

            # Preprocess Images
            for image_name in batch:
                images.append(Image.open(image_name).convert('RGB'))
                self.image_IDs.append(image_name)

            preprocessed_images = []
            for image in images:
                image_input = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
                preprocessed_images.append(image_input)
            preprocessed_images = torch.cat(preprocessed_images)
            del images

            # Build sample
            sample = {"image": preprocessed_images, "text_input": []}

            # Encode Images
            new_encodings = self.model.extract_features(sample, mode="image").image_embeds[:,0,:]
            if self.image_encodings is None:
                self.image_encodings = new_encodings
            else:
                self.image_encodings = torch.cat((self.image_encodings, new_encodings), dim=0)

            self._print_runtime_message(message_type='batch_encoded',batch_num=i+1)

        self.image_encodings = F.normalize(self.image_encodings, p=2, dim=-1)
        # Save if out_file_path specified
        if out_file_path is not None and self.image_encodings is not None:
            with open(out_file_path, 'wb') as f:
                pickle.dump((self.image_IDs, self.image_encodings), f)
            
    def encode_text(self, text):
        # Preprocess text
        text = self.txt_processors["eval"](text)

        # Build sample
        sample = {"image": torch.randn(1, 3, 50, 50), "text_input": [text]}

        # Encode text
        encoded_text = self.model.extract_features(sample, mode="text").text_embeds[0,0,:]
        encoded_text = torch.unsqueeze(encoded_text, dim=0)
        return F.normalize(encoded_text, p=2, dim=-1)