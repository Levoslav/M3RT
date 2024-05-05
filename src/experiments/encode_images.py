import argparse
import os
from src.retrievers.clip_retriever import CLIPRetriever
from src.retrievers.align_retriever import ALIGNRetriever
from src.retrievers.blip2_retriever import BLIP2Retriever
from src.retrievers.openclip_retriever import OpenCLIPRetriever
import sys
import glob

def list_files(directory):
    return glob.glob(os.path.join(directory, "*.jpeg")) + glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.png")) + glob.glob(os.path.join(directory, "*.JPG")) + glob.glob(os.path.join(directory, "*.JPEG"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode images using a specified model.")
    parser.add_argument("dataset", type=str, help="Dataset name", choices=["marine", "photos"], required=True)
    parser.add_argument("model", type=str, help="Model name", choices=["clip", "align", "blip2", "openclip"], required=True)
    parser.add_argument("--version", type=str, help="Version name (only for 'openclip' model)", default=None)

    args = parser.parse_args()
    # encode_images(args.dataset, args.model, args.version)
    
    dataset_path = f'datasets/{args.dataset}/data'
    storage_path = f'saves/image_features'

    if args.model == 'clip':
        retriever = CLIPRetriever()
    elif args.model == 'align':
        retriever = ALIGNRetriever
    elif args.model == 'blip2':
        retriever = BLIP2Retriever
    elif args.model == 'openclip':
        if args.version is None:
            print("Specify openclip --version")
            sys.exit(1)
        else:
            retriever = OpenCLIPRetriever(version=args.version)

    print("encoding dataset...")
    retriever.encode_images(images_paths=list_files(dataset_path), out_dir=storage_path, batch_size=1000)

    