import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import pickle


class Retriever:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_IDs = []
        self.image_encodings = None

    def encode_images(self, images_paths ,out_file_path=None, batch_size=50):
        pass

    def encode_text(self, text):
        pass

    def load_encoded_images(self, file_path):
        # Load list from file
        with open(file_path, 'rb') as f:
            self.image_IDs, self.image_encodings = pickle.load(f)
        self._print_runtime_message(message_type='image_encodings_loaded')

    def compute_cumulation(self, labels: pd.DataFrame, out_file_path=None):
        ranks_cumulation = [0] * len(self.image_IDs)
        for id, label in zip(labels.ID, labels.label):
            ordered_images = self.compare_to_images(label)
            rank = self._find_rank(ordered_images, id)
            ranks_cumulation[rank:] = [x+1 for x in ranks_cumulation[rank:]]

            # Save if out_file_path specified
            if out_file_path is not None :
                with open(out_file_path , 'wb') as f:
                    pickle.dump(ranks_cumulation, f)
        return ranks_cumulation

    def compare_to_images(self, text):  # Returns a sorted list of tuples (cosine_similarity, Image_ID)
        text_encoding = self.encode_text(text).T
        cosine_similarities = torch.mm(self.image_encodings, text_encoding)
        return sorted(zip(cosine_similarities.tolist(),self.image_IDs),reverse=True)

    def plot_top_images(self, sorted_list, n, querry):
        # Create subplots
        fig, axes = plt.subplots(1, n, figsize=(15, 5))

        plt.suptitle("Querry: '" + querry + "'", x=0.1, y=0.95, fontsize=16, ha='left')
        # Display the top n images
        for i in range(min(len(sorted_list), n)):
            _, image_path = sorted_list[i]
            image_path =  image_path
            try:
                # Display the image using matplotlib.image
                img = mpimg.imread(image_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f"Similarity: {round(sorted_list[i][0],4)}")
            except Exception as e:
                print(f"Error displaying image {image_path}: {e}")

        plt.tight_layout()
        plt.show()

    def get_ranks(self, labels: pd.DataFrame):
        ranks = [(label, self._find_rank(self.compare_to_images(label), id))
        for id, label in zip(labels.ID, labels.label)]
        return sorted(ranks)
    
    def _find_rank(self, list_of_tuples, ID):
        for index, (cosine_similarity, Image_ID) in enumerate(list_of_tuples):
            if Image_ID.endswith(ID):
                return index
        return -1  # Return -1 if ID not found in any Image_ID
    
    def _print_runtime_message(self, message_type, batch_num=None, num_of_batches=None, batch_size=None):
        if message_type == 'batch_encoded':
            print(f"    batch {batch_num}. encoded") 
        elif message_type == 'images_encoding_start':
            print(f"Encoding started - batch_size={batch_size} - num_of_batches={num_of_batches}"  )
        elif message_type == 'image_encodings_loaded':
            print("Image encodings(and Image IDs) loaded...")
