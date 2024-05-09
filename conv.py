import pickle

file_name = 'saves/image_features/openclip-ViT_B_16_webli-marine'
with open(file_name, 'rb') as f:
    image_IDs, image_encodings = pickle.load(f)

image_IDs = [x.strip("/").split("/")[-1].split(".")[0] for x in image_IDs]

with open(file_name + ".pkl" , 'wb') as f:
    pickle.dump((image_IDs, image_encodings), f)

with open(file_name+ ".pkl", 'rb') as f:
    image_IDs, image_encodings = pickle.load(f)
    
print(image_IDs)
print(image_encodings.shape)