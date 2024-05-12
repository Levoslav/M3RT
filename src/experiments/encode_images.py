import argparse
import os
import sys
import glob
import timeit

def list_files(directory):
    return glob.glob(os.path.join(directory, "*.jpeg")) + glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.JPG")) + glob.glob(os.path.join(directory, "*.JPEG"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode images using a specified model.")
    parser.add_argument("dataset", type=str, help="Dataset name", choices=["marine", "photos", "marine_compet"])
    parser.add_argument("model", type=str, help="Model name", choices=["clip", "align", "blip2", "openclip"])
    parser.add_argument("version", type=str, help="Version name (only for 'openclip' model)", default=None)

    args = parser.parse_args()
    # encode_images(args.dataset, args.model, args.version)
    
    dataset_path = f'datasets/{args.dataset}/data'
    storage_path = f'saves/image_features/{args.model}-' + ((args.version + '-') if args.model == 'openclip' else '') + args.dataset + '.pkl'

    if args.model == 'clip':
        from clip_retriever import CLIPRetriever
        retriever = CLIPRetriever()
    elif args.model == 'align':
        from align_retriever import ALIGNRetriever
        retriever = ALIGNRetriever()
    elif args.model == 'blip2':
        from blip2_retriever import BLIP2Retriever
        retriever = BLIP2Retriever() 
    elif args.model == 'openclip':
        if args.version is None:
            print("Specify openclip --version")
            sys.exit(1)
        else:
            from openclip_retriever import OpenCLIPRetriever
            retriever = OpenCLIPRetriever(version=args.version)

    batch_size = 100
    if args.dataset == 'marine_compet':
        with open("filename.txt", "r") as file:
            images_paths = [line.rstrip("\n") for line in file]
    else:
        images_paths = list_files(dataset_path)

    print("Encoding dataset...")
    # Measure the execution time and start the encode_images
    execution_time = timeit.timeit(lambda: retriever.encode_images(images_paths, storage_path, batch_size), number=1)
    print(f"Encoding finished, execution time: {execution_time} s")
    