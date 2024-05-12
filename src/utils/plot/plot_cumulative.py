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
        plt.plot(x, y, linestyle='-', linewidth=2, label=name)

    # Set x ticks with stride 5
    stride = m // 10
    plt.xticks(np.arange(1, m + 1, 1), [str(i) if i % stride == 0 or i == 1 else '' for i in range(1, m + 1)])

    # Set y ticks with stride 10
    plt.yticks(np.arange(1, 101, 1), [str(i) if i % 10 == 0 or i == 1 else '' for i in range(1, 101)])

    # Add vertical lines as background for better orientation
    # plt.grid(axis='x', linestyle='--', alpha=0.5)

    # Labeling axes
    plt.xlabel('Rank')
    plt.ylabel('Count')

    # Title
    plt.title(title)

    # Legend
    plt.legend()

    # Show plot
    plt.show()

directory = "saves/cumulations/"
marine_short_cumulations = ['openclip-ViT_SO400M_14_webli-marine-short.pkl', 'openclip-ViT_H_14_dfn5b-marine-short.pkl', 'openclip-ViT_L_16_webli-marine-short.pkl', 'openclip-ViT_B_16_webli-marine-short.pkl', 'openclip-ViT_G_14_laion2b-marine-short.pkl', 'openclip-ViT_H_14_laion2b-marine-short.pkl']
marine_long_cumulations = ['openclip-ViT_SO400M_14_webli-marine-long.pkl', 'openclip-ViT_H_14_dfn5b-marine-long.pkl', 'openclip-ViT_L_16_webli-marine-long.pkl', 'openclip-ViT_B_16_webli-marine-long.pkl', 'openclip-ViT_G_14_laion2b-marine-long.pkl', 'openclip-ViT_H_14_laion2b-marine-long.pkl']
photos_short_cumulations = ['openclip-ViT_SO400M_14_webli-photos-short.pkl', 'openclip-ViT_H_14_dfn5b-photos-short.pkl', 'openclip-ViT_L_16_webli-photos-short.pkl', 'openclip-ViT_B_16_webli-photos-short.pkl', 'openclip-ViT_G_14_laion2b-photos-short.pkl', 'openclip-ViT_H_14_laion2b-photos-short.pkl']
photos_long_cumulations = ['openclip-ViT_SO400M_14_webli-photos-long.pkl', 'openclip-ViT_H_14_dfn5b-photos-long.pkl', 'openclip-ViT_L_16_webli-photos-long.pkl', 'openclip-ViT_B_16_webli-photos-long.pkl', 'openclip-ViT_G_14_laion2b-photos-long.pkl', 'openclip-ViT_H_14_laion2b-photos-long.pkl']
names = ['ViT_SO400M_14_webli', 'ViT_H_14_dfn5b', 'ViT_L_16_webli', 'ViT_B_16_webli', 'ViT_G_14_laion2b (the OpenClip I used before)', 'ViT_H_14_laion2b (the OpenClip used in competition)']

cumulations = [load_from_file(directory+file_name) for file_name in marine_long_cumulations]

plot_cumulative_graph(cumulations, 100 , names, " Marine dataset  /  long labels  /  Open Clips")