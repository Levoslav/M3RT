import argparse
import os
import sys
import glob
import timeit
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode images using a specified model.")
    parser.add_argument("dataset", type=str, help="Dataset name", choices=["marine", "photos", "lsc"])
    parser.add_argument("labels", type=str, help="Type of labels", choices=["short", "long"])
    parser.add_argument("model", type=str, help="Model name", choices=["clip", "align", "blip2", "openclip"])
    parser.add_argument("version", type=str, help="Version name (only for 'openclip' model)", default=None)


    args = parser.parse_args()

    image_encodings_path = f'saves/image_features/{args.model}-' + ((args.version + '-') if args.model == 'openclip' else '') + args.dataset + '.pkl'
    labels_path = f'datasets/{args.dataset}/labels/labels.csv'
    cumulation_storage_path = f'saves/cumulations/{args.model}-' + ((args.version + '-') if args.model == 'openclip' else '') + args.dataset + f'-{args.labels}.pkl'

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

    # Load image_features
    retriever.load_encoded_images(image_encodings_path)

    # Load and prepare labels
    labels = pd.read_csv(labels_path, dtype={"ID": str})
    if args.labels == 'short':
        labels = labels[['ID','short_label']]
    elif args.labels == 'long':
        labels = labels[['ID','long_label']]
    labels.columns = ['ID', 'label']

    print("Starting evaluation")
    execution_time = timeit.timeit(lambda: retriever.compute_cumulation(labels, cumulation_storage_path), number=1)
    print(f"Evaluation finished, execution time: {execution_time} s")

    
