import numpy as np
import matplotlib.pyplot as plt
import pickle

def load_from_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def plot_cumulative_graph(cumulations, m, names, title):
    # Generate x values (ranks)
    x = np.arange(1, m + 1)
    
    # Plot each set of cumulations
    for cumulation, name in zip(cumulations, names):
        cumulation = cumulation[:m]
        y = np.array(cumulation)
        plt.plot(x, y, linestyle='-', linewidth=1.5, label=name)

    # Set ticks
    stride = m // 10 # 10 ticks
    ticks = [1] + list(range(stride ,m+1, stride))
    plt.xticks(ticks)
    plt.yticks(range(0,101,10))

    # Set grids
    plt.grid(axis='y', linestyle='--', alpha=0.25)
    plt.grid(axis='x', linestyle='--', alpha=0.25)

    # Labeling axes
    plt.xlabel('Rank')
    plt.ylabel('Count')

    # Title
    plt.title(title)

    # Legend
    plt.legend()

    # Show plot
    plt.show()
    
if __name__ == "__main__":
    directory = "saves/cumulations/"
    marine_short_cumulations = ['openclip-ViT_SO400M_14_webli-marine-short.pkl', 'openclip-ViT_H_14_dfn5b-marine-short.pkl', 'openclip-ViT_L_16_webli-marine-short.pkl', 'openclip-ViT_B_16_webli-marine-short.pkl', 'openclip-ViT_G_14_laion2b-marine-short.pkl', 'openclip-ViT_H_14_laion2b-marine-short.pkl']
    marine_long_cumulations = ['openclip-ViT_SO400M_14_webli-marine-long.pkl', 'openclip-ViT_H_14_dfn5b-marine-long.pkl', 'openclip-ViT_L_16_webli-marine-long.pkl', 'openclip-ViT_B_16_webli-marine-long.pkl', 'openclip-ViT_G_14_laion2b-marine-long.pkl', 'openclip-ViT_H_14_laion2b-marine-long.pkl']
    photos_short_cumulations = ['openclip-ViT_SO400M_14_webli-photos-short.pkl', 'openclip-ViT_H_14_dfn5b-photos-short.pkl', 'openclip-ViT_L_16_webli-photos-short.pkl', 'openclip-ViT_B_16_webli-photos-short.pkl', 'openclip-ViT_G_14_laion2b-photos-short.pkl', 'openclip-ViT_H_14_laion2b-photos-short.pkl']
    photos_long_cumulations = ['openclip-ViT_SO400M_14_webli-photos-long.pkl', 'openclip-ViT_H_14_dfn5b-photos-long.pkl', 'openclip-ViT_L_16_webli-photos-long.pkl', 'openclip-ViT_B_16_webli-photos-long.pkl', 'openclip-ViT_G_14_laion2b-photos-long.pkl', 'openclip-ViT_H_14_laion2b-photos-long.pkl']
    names = ['ViT_SO400M_14_webli', 'ViT_H_14_dfn5b', 'ViT_L_16_webli', 'ViT_B_16_webli', 'ViT_G_14_laion2b (the OpenClip I used before)', 'ViT_H_14_laion2b (the OpenClip used in competition)']

    cumulations = [load_from_file(directory+file_name) for file_name in photos_short_cumulations]

    plot_cumulative_graph(cumulations, 50 , names, " Photos dataset  |  short labels  |  Open Clips")