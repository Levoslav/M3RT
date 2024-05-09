import pickle

file_dir =  'saves/image_features/'
names = ['openclip-ViT_B_16_webli-marine','openclip-ViT_B_16_webli-photos','openclip-ViT_G_14_laion2b-marine','openclip-ViT_G_14_laion2b-photos', 'openclip-ViT_H_14_dfn5b-marine', 'openclip-ViT_H_14_dfn5b-photos', 'openclip-ViT_H_14_laion2b-marine', 'openclip-ViT_H_14_laion2b-photos', 'openclip-ViT_L_16_webli-marine', 'openclip-ViT_L_16_webli-photos', 'openclip-ViT_SO400M_14_webli-marine', 'openclip-ViT_SO400M_14_webli-photos']

for name in names: 
    with open(file_dir + name, 'rb') as f:
        image_IDs, image_encodings = pickle.load(f)

    image_IDs = [f'{int(x.strip("/").split("/")[-1].split(".")[0]):05d}' for x in image_IDs]

    with open(file_dir + name + ".pkl" , 'wb') as f:
        pickle.dump((image_IDs, image_encodings), f)

    print(f'{name} - {image_encodings.shape}')

