from retriever import Retriever
import os
import torch
import torch.nn.functional as F
from PIL import Image
import requests
import pickle
import pandas as pd
from transformers import AlignProcessor, AlignModel

class ALIGNRetriever(Retriever):
    def __init__(self) -> None:
        super().__init__()
        self.preprocess = AlignProcessor.from_pretrained("kakaobrain/align-base")
        self.model = AlignModel.from_pretrained("kakaobrain/align-base")

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

            preprocessed_input = self.preprocess(text="",images=images, return_tensors="pt")
            del images

            # Encode Images
            with torch.no_grad():
                outputs = self.model(**preprocessed_input)
                if self.image_encodings is None:
                    self.image_encodings = outputs.image_embeds
                else:
                    self.image_encodings = torch.cat((self.image_encodings, outputs.image_embeds), dim=0)

        self.image_encodings = F.normalize(self.image_encodings, p=2, dim=-1)
        # Save if out_dir specified
        if out_dir is not None and self.image_encodings is not None:
            torch.save(self.image_encodings, out_dir + '/ALIGN_image_encodings.pth')
            with open(out_dir + '/ALIGN_images_IDs.pkl', 'wb') as f:
                pickle.dump(self.image_IDs, f)

    def encode_text(self, text):
        # Preprocess text
        there_must_be_picture = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)  # I don't like it either
        preprocessed_input = self.preprocess(text=text,images=there_must_be_picture, return_tensors="pt")

        # Encode text
        outputs = self.model(**preprocessed_input)
        return F.normalize(outputs.text_embeds, p=2, dim=-1)

    def load_encoded_images(self, directory):
        # Load tensor from file
        self.image_encodings = torch.load(os.path.join(directory ,'ALIGN_image_encodings.pth'))
        # Load list from file
        with open(os.path.join(directory , 'ALIGN_images_IDs.pkl'), 'rb') as f:
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
                dir = os.path.join(out_dir , 'ALIGN_cumulation.pkl')
                with open(dir , 'wb') as f:
                    pickle.dump(ranks_cumulation, f)
        return ranks_cumulation