import pickle

file_name = 'saves/image_features/openclip-ViT_B_16_webli-marine'
with open(file_name, 'rb') as f:
    image_IDs, image_encodings = pickle.load(f)

print(image_IDs)
print(image_encodings.shape)