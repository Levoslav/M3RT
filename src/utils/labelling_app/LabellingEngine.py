import os
import sys
import random

class LabellingEngine:
    def __init__(self, dataset_dir, output_csv_path, seed,
                model_assigned_labels=None,
                nounlist=None, 
                short_and_long_labels=True) -> None:
        
        self.dataset_dir = dataset_dir
        self.output_csv_path = output_csv_path
        self.seed = seed
        self.model_assigned_labels = model_assigned_labels
        self.nounlist = nounlist
        self.short_and_long_labels = short_and_long_labels
        self.used_ids = set()
        random.seed(self.seed)

        # Check, whether provided output_csv_file exists
        if os.path.exists(output_csv_path):
            with open(output_csv_path, "r", encoding='utf-8') as file:
                first_line = file.readline().strip()
                if first_line == "ID,short_label,long_label":
                    self.short_and_long_labels = True
                    for line in file:
                        line = line.strip()
                        if line == "":
                            break
                        id, short_label, long_label = line.split(",")
                        short_label = short_label.strip('"')
                        long_label = long_label.strip('"')
                        self.used_ids.add(id)
                         
                elif first_line == "ID,label":
                    self.short_and_long_labels = False
                    for line in file:
                        line = line.strip()
                        if line == "":
                            break
                        id, label = line.strip().split(",")
                        label = label.strip('"')
                        self.used_ids.add(id)
                else:
                    print(f"Couldn't recognise provided csv file: {output_csv_path} structure")
                    sys.exit(1)
        else:
            # Create the file
            with open(output_csv_path, "w", encoding='utf-8') as file:
                if short_and_long_labels:
                    file.write("ID,short_label,long_label\n")
                else:
                    file.write("ID,label\n")

        # Get list of image filenames in the directory
        self.image_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
        # Extract IDs from filenames
        self.image_ids = [int(f.split('.')[0]) for f in self.image_files]
        self.max_id = max(self.image_ids)

    def get_next_id(self):
        while True:
            random_id = f"{random.randint(1, self.max_id):05d}"
            if random_id not in self.used_ids:
                self.used_ids.add(random_id)
                return random_id

    def get_next_record(self): # Record = (image_id, image_path, model_generated_labels_of_image)
        image_id = self.get_next_id()
        image_path = os.path.join(self.dataset_dir, f"{image_id}.jpg")
        if self.model_assigned_labels is None or self.nounlist is None:
            model_generated_labels_of_image = None
        else:
            model_generated_labels_of_image = []
            for generated_label_id in self.model_assigned_labels[int(image_id)]:
                model_generated_labels_of_image.append(self.nounlist[int(generated_label_id)])
        
        return image_id, image_path, model_generated_labels_of_image

    def csv_append(self, ID, short_label, long_label=None):
        if long_label is None:
            with open(self.output_csv_path, "a", encoding='utf-8') as file:
                file.write(f'{ID},"{short_label}"\n')
        else:
            with open(self.output_csv_path, "a", encoding='utf-8') as file:
                file.write(f'{ID},"{short_label}","{long_label}"\n')

# labeler = LabellingEngine("dataset", 69, 'labels.csv', None, None, short_and_long_labels=True)
# labeler.csv_append(ID='00006',short_label="turtle",long_label="turtle chilling on a sun")
# print(labeler.used_ids)