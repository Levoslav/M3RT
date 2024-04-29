import tkinter as tk
from PIL import Image, ImageTk
from LabellingEngine import LabellingEngine
import pickle
import torch

class LabellingApp:
    def __init__(self, root, dataset_dir, output_csv_path, seed, 
                model_assigned_labels=None,
                nounlist=None, 
                short_and_long_labels=True):
        
        self.engine = LabellingEngine(dataset_dir,  
                                      output_csv_path, 
                                      seed,
                                      model_assigned_labels, 
                                      nounlist,
                                      short_and_long_labels)
        
        self.show_labels = model_assigned_labels is not None and nounlist is not None
        self.root = root
        self.root.title("Turbo Labeler 3000")
        self.create_widgets()

    def create_widgets(self):
        # Frame for picture
        self.picture_frame = tk.Frame(self.root)
        self.picture_frame.grid(row=0, column=0, sticky="nsew")
        
        # Display model generated labels only if provided
        if self.show_labels:
            # Frame for labels
            self.label_frame = tk.Frame(self.root)
            self.label_frame.grid(row=0, column=1, sticky="nsew")

            # Right top corner: Display labels
            self.labels = []
            for i in range(10):
                label = tk.Label(self.label_frame, text="")
                label.grid(row=i, column=0, sticky="w")
                self.labels.append(label)
        
        # Frame for buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.grid(row=1, column=1, sticky="nsew")

        # Frame for text fields
        self.entry_frame = tk.Frame(self.root)
        self.entry_frame.grid(row=1, column=0, sticky="nsew")
        
        # Left top corner: Display picture
        self.image_id_label = tk.Label(self.picture_frame, text="Image ID")
        self.image_id_label.grid(row=0)
        self.image_label = tk.Label(self.picture_frame)
        self.image_label.grid(row=1, column=0)

        # Right lower corner: Text fields and buttons

        if self.engine.short_and_long_labels:
            self.text_1 = tk.Text(self.entry_frame, height=3, width=30, wrap=tk.WORD)
            self.text_1_title = tk.Label(self.entry_frame, text="Short label:")
            self.text_1.grid(row=1, column=0, padx=5, pady=5)
            self.text_1_title.grid(row=0, column=0, padx=5, pady=5)

            self.text_2 = tk.Text(self.entry_frame,height=3, width=30, wrap=tk.WORD)
            self.text_2_title = tk.Label(self.entry_frame, text="Long label:")
            self.text_2.grid(row=1, column=1, padx=5, pady=5)
            self.text_2_title.grid(row=0, column=1, padx=5, pady=5)
        else:
            self.text_1 = tk.Text(self.entry_frame, height=3, width=30, wrap=tk.WORD)
            self.text_1_title = tk.Label(self.entry_frame, text="Label:")
            self.text_1.grid(row=1, column=0, padx=5, pady=5)
            self.text_1_title.grid(row=0, column=0, padx=5, pady=5)
        
        self.button1 = tk.Button(self.button_frame, text="Save & Next", height=3, width=15, command=self.next_and_save)
        self.button1.grid(row=0, column=2, padx=5, pady=5)
        self.button2 = tk.Button(self.button_frame, text="Ain't labelling that", command=self.refresh)
        self.button2.grid(row=1, column=2, padx=5, pady=5)
        
    def next_and_save(self):
        # Read Labels from textfields
        if self.engine.short_and_long_labels:
            short_label = self.text_1.get("1.0", tk.END).replace('\n', ' ').strip()
            long_label = self.text_2.get("1.0", tk.END).replace('\n', ' ').strip()
        else:
            label = self.text_1.get("1.0", tk.END).replace('\n', ' ').strip()
        
        # Check whether we filled all text fields 
        if self.engine.short_and_long_labels:
            if short_label == "" and long_label == "":
                self.text_1.configure(highlightbackground="red")
                self.text_2.configure(highlightbackground="red")
                return
            elif short_label == "":
                self.text_1.configure(highlightbackground="red")
                self.text_2.configure(highlightbackground="white")
                return
            elif long_label == "":
                self.text_2.configure(highlightbackground="red")
                self.text_1.configure(highlightbackground="white")
                return
        else:
            if label == "":
                self.text_1.configure(highlightbackground="red")
                return
            
        # Update output csv
        id = self.image_id_label.cget("text").strip()
        if self.engine.short_and_long_labels:
            self.engine.csv_append(id, short_label, long_label)
        else:
            self.engine.csv_append(id, label)

        # Get new image record and update page
        image_id, image_path, model_generated_labels_of_image = self.engine.get_next_record()
        self.display_page(image_id, image_path, model_generated_labels_of_image)
    
    def refresh(self):
        # Get new image record and update page
        image_id, image_path, model_generated_labels_of_image = self.engine.get_next_record()
        self.display_page(image_id, image_path, model_generated_labels_of_image)
    
    def display_picture(self, path, height=int(180*1.4), width=int(320*1.4)):
        # Function to display picture
        image = Image.open(path)
        image = image.resize((width, height))  # Resize the image as needed
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo  # Keep a reference to avoid garbage collection
    
    def display_labels(self, labels):
        # Function to display labels
        for i, label_text in enumerate(labels):
            if i < len(self.labels):
                self.labels[i].config(text=label_text)

    def display_page(self, image_id, image_path, model_generated_labels_of_image=None):
        # Set edges color
        self.text_1.configure(highlightbackground="white")
        if self.engine.short_and_long_labels:
            self.text_2.configure(highlightbackground="white")

        # Reset input fields
        if self.engine.short_and_long_labels:
            self.text_1.delete(1.0, tk.END)
            self.text_2.delete(1.0, tk.END)
        else:
            self.text_1.delete(1.0, tk.END)
        
        # Display now image_id, image and optionally labels
        self.image_id_label.configure(text=image_id)
        self.display_picture(image_path)
        if self.show_labels:
            self.display_labels(model_generated_labels_of_image)

    def run(self):
        print(self.engine.used_ids)
        self.refresh() # Pick first image
        self.root.mainloop()
        
def load_from_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    

# Example usage:
if __name__ == "__main__":
    root = tk.Tk()
    dataset_path = "datasets/marine/data"
    output_csv_path = "datasets/marine/labels/labels1.csv"
    other_files_path = "datasets/marine/other_files/"

    nounlist = load_from_file(other_files_path + "nounlist.pkl")
    model_assigned_labels = torch.load(other_files_path + "marine_results.pth")
    
    app = LabellingApp(root, 
                       dataset_path,  
                       output_csv_path, 
                       69, 
                       model_assigned_labels, 
                       nounlist, 
                       short_and_long_labels=True)
    app.run()
