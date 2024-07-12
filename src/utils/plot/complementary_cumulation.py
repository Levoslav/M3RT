from itertools import combinations
import  pickle
from plot_cumulative import plot_cumulative_graph
import argparse

def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def compute_cumulation(ranks, max_size = 9566):
    ranks_cumulation = [0] * max_size
    for rank in ranks:
        ranks_cumulation[rank:] = [x+1 for x in ranks_cumulation[rank:]]
    return ranks_cumulation
    
def plot_cumulative_complementary_graphs(models_results: dict, models_colors: dict , out_dir=None, n=500):
    # We want to save this plots in grid under diagonal so thats why reversed()
    names = reversed(models_results.keys())
    out_file_path = None
    for name1, name2 in list(combinations(names, 2)):
        if out_dir is not None:
            out_file_path = out_dir + name1 + "-" + name2 + ".png"
            plot_single_complementary_graph(models_results[name1], models_results[name2],
                                            name1, name2, models_colors[name1], models_colors[name2], n, out_file_path )
       

def plot_single_complementary_graph(ranks_1, ranks_2, name_1, name_2, color_1, color_2, n=500, out_file_path=None):
    complementary_ranks = [min(r1,r2) for r1, r2 in zip(ranks_1, ranks_2)]
    cumulations = [compute_cumulation(ranks_1), compute_cumulation(ranks_2), compute_cumulation(complementary_ranks)]
    plot_cumulative_graph(cumulations, n , [name_1, name_2, "Complement"], None, colors=[color_1, color_2, "dimgrey"],out_path=out_file_path, line_width=2.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode images using a specified model.")
    parser.add_argument("--dataset",default="lsc", type=str, help="Dataset name", choices=["marine", "photos", "marine_compet", "lsc"])
    parser.add_argument("--labels",default="long", type=str, help="Type of labels", choices=["short", "long"])
    parser.add_argument("--tight_layout", default=True, type=bool, help="True if no title and axis labels should be displayed")
    parser.add_argument("--interactive", default=False, type=bool, help="True if labels should show if you point at them")
    parser.add_argument("--models", default=['openclip-ViT_SO400M_14_webli','openclip-ViT_L_16_webli','openclip-ViT_B_16_webli','openclip-ViT_H_14_dfn5b','openclip-ViT_G_14_laion2b'],
                        type=list, help="Selected models ['clip', 'blip2', 'align', 'openclip-ViT_SO400M_14_webli','openclip-ViT_L_16_webli','openclip-ViT_B_16_webli','openclip-ViT_H_14_dfn5b','openclip-ViT_G_14_laion2b','openclip-ViT_H_14_laion2b'  ]")
    args = parser.parse_args()
    
    path = "saves/ranks/"
    # Create d input in form of dictionary
    d = {version.split("-")[-1]:[b for a,b in load_from_file(path+f"{version}-{args.dataset}-{args.labels}.pkl")] 
            for version in args.models}
    
    colors_d = {"clip":"lightseagreen", 
                "blip2":"purple",
                "align":"gold",
                "ViT_SO400M_14_webli":"cornflowerblue",
                "ViT_L_16_webli":"green",
                "ViT_B_16_webli":"red",
                "ViT_H_14_dfn5b":"darkgrey",
                "ViT_G_14_laion2b":"orange",
                "ViT_H_14_laion2b":"orchid"}
    
        
    plot_cumulative_complementary_graphs(d,colors_d, out_dir=f"saves/plots/cumulative_complementary_graphs/{args.dataset}_{args.labels}/", n= 50 if args.dataset == "photos" else 500)
    