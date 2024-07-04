import matplotlib.pyplot as plt
# import numpy as np
from itertools import combinations
import os
import pickle
import pandas as pd
from matplotlib.lines import Line2D

def plot_ranks2D(models_results: dict, labels, out_dir=None, logscale=True ,n=None):
    names = models_results.keys()
    out_file_path = None
    for name1, name2 in list(combinations(names, 2)):
        if out_dir is not None:
            out_file_path = out_dir + name1 + "-" + name2 + ".png"
        if logscale:
            create_ranks2D_plot([x+1 for x in models_results[name1]], [x+1 for x in models_results[name2]],
                            name1, name2, labels, out_file_path, logscale, n)
        else:
            create_ranks2D_plot(models_results[name1], models_results[name2],
                            name1, name2, labels, out_file_path, logscale, n)

def create_ranks2D_plot(x_axis, y_axis, name_x, name_y, labels, out_file_path=None, logscale=False, n=None):
    # Get color for each point, based on the distance from the diagonal
    colors = []
    for x, y in zip(x_axis, y_axis):
        if y > x:
            colors.append('red')
        elif y < x:
            colors.append('green')
        else:
            colors.append('orange')

    # Scatter points
    # norm = plt.Normalize(1,4)
    # cmap = plt.cm.RdYlGn
    fig,ax = plt.subplots()
    sc = plt.scatter(x_axis, y_axis, marker='o', c=colors)

    annot = ax.annotate("", xy=(0,0), xytext=(10,10),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"))
    annot.set_visible(False)


    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = " ".join([labels[n] for n in ind["ind"]])
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('grey')
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    # Plot the diagonal line
    plt.plot([1, 10000], [1, 10000], color='gray', linestyle='--', linewidth=0.5)

    plt.title(f"{name_x}   vs   {name_y}")
    plt.xlabel(f"{name_x} rank {'(logscale)' if logscale else ''}")
    plt.ylabel(f"{name_y} rank {'(logscale)' if logscale else ''}")

    if logscale:
        axx = plt.gca()
        axx.set_xscale('log')
        axx.set_yscale('log')

    # plt.xlim(0,10)
    # plt.ylim(0,10)

    legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=f'({colors.count("red")}) "{name_x}" better'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label=f'({colors.count("green")}) "{name_y}" better'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label=f'({colors.count("orange")}) equally good')
    ]

    # Add legend to plot
    plt.legend(handles=legend_elements)

    # Save if path specified
    if out_file_path is not None:
        plt.savefig(out_file_path)

    # Show the plot
    plt.show()

def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
if __name__ == "__main__":
    dataset = "photos"
    labels = "long"
    models = "openclips"
    df_labels = pd.read_csv(f'datasets/{dataset}/labels/labels.csv')
    labels_list = sorted(df_labels[f'{labels}_label'].to_list())
    if models == "openclips":
        versions = ["ViT_SO400M_14_webli",
                    "ViT_H_14_dfn5b",
                    "ViT_L_16_webli",
                    "ViT_B_16_webli",
                    "ViT_G_14_laion2b",
                    "ViT_H_14_laion2b"]
        path = "saves/ranks/"
        d = {version:[b for a,b in load_from_file(path+f"openclip-{version}-{dataset}-{labels}.pkl")] 
              for version in versions}
        
    plot_ranks2D(d,labels_list, out_dir=f"saves/plots/plot2D/{dataset}_{labels}/")