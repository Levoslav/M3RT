import argparse
import pickle
import torch
import pandas as pd

def compute_cosine_distances(query_id, candidate1_id, candidate2_id):
    if query_id not in ID_index or candidate1_id not in ID_index or candidate2_id not in ID_index:
        return 1.11, 1.11, 1.11
    query = image_encodings[ID_index[query_id]]
    candidate1 = image_encodings[ID_index[candidate1_id]]
    candidate2 = image_encodings[ID_index[candidate2_id]]
    q_c1_cosd = 1 - float(torch.mm(query.unsqueeze(0).float(), candidate1.unsqueeze(1).float())[0,0])
    q_c2_cosd = 1 - float(torch.mm(query.unsqueeze(0).float(), candidate2.unsqueeze(1).float())[0,0])
    c1_c2_cosd = 1 - float(torch.mm(candidate1.unsqueeze(0).float(), candidate2.unsqueeze(1).float())[0,0])
    return q_c1_cosd, q_c2_cosd, c1_c2_cosd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode images using a specified model.")
    parser.add_argument("old_csv", type=str, help="old csv name")
    parser.add_argument("extended_csv", type=str, help="New extended csv name")
    parser.add_argument("model", type=str, help="Model name", choices=["ViT_SO400M_14_webli", "ViT_H_14_dfn5b", "ViT_H_14_laion2b","clip"])
    args = parser.parse_args()
 
    csv_path = f'saves/reset_csvs/'
    encodings_path = f'saves/image_features/openclip-{args.model}-reset.pkl'
    if args.model == "clip":
        encodings_path = f'saves/image_features/clip-reset.pkl'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    number_of_droped_lines = 0

    print("Loading encodings...")
    with open(encodings_path, 'rb') as f:
        image_IDs, image_encodings = pickle.load(f)
    
    print("Indexing encodings IDs...")
    ID_index = {id:i for i,id in enumerate(image_IDs)}

    print(f"Moving encodings to {device}...")
    image_encodings.to(device)

    print("Loading old csv")
    df = pd.read_csv(csv_path+args.old_csv)
    print("size: ", df.shape)

    print("Starting to compute distances...")
    q_c1_cosd = []
    q_c2_cosd = []
    c1_c2_cosd = []
    counter = 1
    for i, row in df[['candidate1_path','candidate2_path','target_path']].iterrows():
        c1_id = row['candidate1_path'].split("/")[1].split(".")[0]
        c2_id = row['candidate2_path'].split("/")[1].split(".")[0]
        q_id = row['target_path'].split("/")[1].split(".")[0]
        
        q_c1_distance, q_c2_distance, c1_c2_distance = compute_cosine_distances(q_id, c1_id, c2_id) 

        q_c1_cosd.append(q_c1_distance)
        q_c2_cosd.append(q_c2_distance)
        c1_c2_cosd.append(c1_c2_distance)

        counter += 1
        if counter % 100 == 0: # Print message every 100 rows
            print(f"epoch {int(counter/100)}")
    print(f"Distances computet. Skiped lines: {number_of_droped_lines}")

    print("Extending and saving new csv...")
    name1 = f"{args.model}_cosdist_query_candidate1"
    name2 = f"{args.model}_cosdist_query_candidate2"
    name3 = f"{args.model}_cosdist_candidates"
    df[name1] = q_c1_cosd
    df[name2] = q_c2_cosd
    df[name3] = c1_c2_cosd
    print("New size: ", df.shape)
    df.to_csv(csv_path+args.extended_csv,index=False)
    print(f"Saved: {args.extended_csv}")
    