import  pickle
from plot_cumulative import plot_cumulative_graph

def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def compute_cumulation(ranks):
    ranks_cumulation = [0] * 9412
    for rank in ranks:
        ranks_cumulation[rank:] = [x+1 for x in ranks_cumulation[rank:]]
    return ranks_cumulation
    

# "ViT_H_14_dfn5b",
# "ViT_L_16_webli",
# "ViT_G_14_laion2b",
# "ViT_H_14_laion2b"
path = "saves/ranks/"
dataset = "marine"
labels = "short"
name_1 = "ViT_SO400M_14_webli"
name_2 =  "ViT_B_16_webli"

ranks_1 = [b for a,b in load_from_file(path+f"openclip-{name_1}-{dataset}-{labels}.pkl")] 
ranks_2 = [b for a,b in load_from_file(path+f"openclip-{name_2}-{dataset}-{labels}.pkl")]
print(len(ranks_1))
print(len(ranks_2))
comb_ranks = [min(r1,r2) for r1, r2 in zip(ranks_1, ranks_2)]

cumulations = [compute_cumulation(ranks_1), compute_cumulation(ranks_2), compute_cumulation(comb_ranks)]

plot_cumulative_graph(cumulations, 1000 , [name_1, name_2, "Combination"], " Marine dataset  |  short labels ")
