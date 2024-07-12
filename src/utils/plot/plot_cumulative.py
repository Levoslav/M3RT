import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_from_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_cumulative_graph(cumulations, m, names, title , colors, out_path, line_width):
    # Generate x values (ranks)
    x = np.arange(1, m + 1)
    
    # Plot each set of cumulations
    for cumulation, (name ,color) in zip(cumulations, zip(names,colors)):
        cumulation = cumulation[:m]
        y = np.array(cumulation)
        plt.plot(x, y, linestyle='-', linewidth=line_width, label=name, color=color, alpha=0.8)

    # Set ticks
    stride = m // 10 # 10 ticks
    ticks = [1] + list(range(stride ,m+1, stride))
    plt.xticks(ticks)
    plt.yticks(range(0,101,10))

    # Set grids
    plt.grid(axis='y', linestyle='--', alpha=0.25)
    plt.grid(axis='x', linestyle='--', alpha=0.25)

    # Labeling axes
    # plt.xlabel('Rank')
    # plt.ylabel('Count')

    # Title
    if title is not None:
        plt.title(title)

    # Legend
    plt.legend(fontsize='13')

    # Show and save plot
    plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.show()
    
if __name__ == "__main__":
    directory = "saves/cumulations/"
    marine_short_cumulations = ['clip-marine-short.pkl','blip2-marine-short.pkl','align-marine-short.pkl','openclip-ViT_SO400M_14_webli-marine-short.pkl', 'openclip-ViT_H_14_dfn5b-marine-short.pkl', 'openclip-ViT_L_16_webli-marine-short.pkl', 'openclip-ViT_B_16_webli-marine-short.pkl', 'openclip-ViT_G_14_laion2b-marine-short.pkl', 'openclip-ViT_H_14_laion2b-marine-short.pkl']
    marine_long_cumulations = ['clip-marine-long.pkl','blip2-marine-long.pkl','align-marine-long.pkl','openclip-ViT_SO400M_14_webli-marine-long.pkl', 'openclip-ViT_H_14_dfn5b-marine-long.pkl', 'openclip-ViT_L_16_webli-marine-long.pkl', 'openclip-ViT_B_16_webli-marine-long.pkl', 'openclip-ViT_G_14_laion2b-marine-long.pkl', 'openclip-ViT_H_14_laion2b-marine-long.pkl']
    photos_short_cumulations = ['clip-photos-short.pkl','blip2-photos-short.pkl','align-photos-short.pkl','openclip-ViT_SO400M_14_webli-photos-short.pkl', 'openclip-ViT_H_14_dfn5b-photos-short.pkl', 'openclip-ViT_L_16_webli-photos-short.pkl', 'openclip-ViT_B_16_webli-photos-short.pkl', 'openclip-ViT_G_14_laion2b-photos-short.pkl', 'openclip-ViT_H_14_laion2b-photos-short.pkl']
    photos_long_cumulations = ['clip-photos-long.pkl','blip2-photos-long.pkl','align-photos-long.pkl','openclip-ViT_SO400M_14_webli-photos-long.pkl', 'openclip-ViT_H_14_dfn5b-photos-long.pkl', 'openclip-ViT_L_16_webli-photos-long.pkl', 'openclip-ViT_B_16_webli-photos-long.pkl', 'openclip-ViT_G_14_laion2b-photos-long.pkl', 'openclip-ViT_H_14_laion2b-photos-long.pkl']
    lsc_short_cumulations = ['clip-lsc-short.pkl','blip2-lsc-short.pkl','align-lsc-short.pkl','openclip-ViT_SO400M_14_webli-lsc-short.pkl', 'openclip-ViT_H_14_dfn5b-lsc-short.pkl', 'openclip-ViT_L_16_webli-lsc-short.pkl', 'openclip-ViT_B_16_webli-lsc-short.pkl', 'openclip-ViT_G_14_laion2b-lsc-short.pkl', 'openclip-ViT_H_14_laion2b-lsc-short.pkl']
    lsc_long_cumulations = ['clip-lsc-long.pkl','blip2-lsc-long.pkl','align-lsc-long.pkl','openclip-ViT_SO400M_14_webli-lsc-long.pkl', 'openclip-ViT_H_14_dfn5b-lsc-long.pkl', 'openclip-ViT_L_16_webli-lsc-long.pkl', 'openclip-ViT_B_16_webli-lsc-long.pkl', 'openclip-ViT_G_14_laion2b-lsc-long.pkl', 'openclip-ViT_H_14_laion2b-lsc-long.pkl']
    
    names = ['CLIP_B_32','BLIP-2','ALIGN','ViT_SO400M_14_webli','ViT_L_16_webli' ,'ViT_B_16_webli' , 'ViT_H_14_dfn5b',  'ViT_G_14_laion2b', 'ViT_H_14_laion2b']
    colors = [
    "lightseagreen",
    "purple",
    "gold",
    "cornflowerblue",
    "green",
    "red",
    "darkgrey",
    "orange",
    "orchid"
    ]

    dataset = "lsc"
    labels = "long"
    cumulations = None
    head_size = 500

    if dataset == "photos":
        head_size = 50
        if labels == "short":
            cumulations = photos_short_cumulations
        elif labels == "long":
            cumulations = photos_long_cumulations
    elif dataset == "marine":
        head_size = 500
        if labels == "short":
            cumulations = marine_short_cumulations
        elif labels == "long":
            cumulations = marine_long_cumulations
    elif dataset == "lsc":
        head_size = 500
        if labels == "short":
            cumulations = lsc_short_cumulations
        elif labels == "long":
            cumulations = lsc_long_cumulations

    cumulations = [load_from_file(directory+file_name) for file_name in cumulations]

    out_path = f"saves/plots/cumulative_graphs/{dataset}_{labels}.png"
    plot_cumulative_graph(cumulations, head_size , names, None, colors, out_path,1.5)