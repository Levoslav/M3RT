import matplotlib.pyplot as plt
# import numpy as np
from itertools import combinations
import os
import pickle
import pandas as pd
from matplotlib.lines import Line2D
import argparse

def plot_ranks2D(models_results: dict, models_colors: dict ,labels, out_dir=None, logscale=True ,n=None):
    names = models_results.keys()
    out_file_path = None
    for name1, name2 in list(combinations(names, 2)):
        if out_dir is not None:
            out_file_path = out_dir + name1 + "-" + name2 + ".png"
        if logscale:
            create_ranks2D_plot([x+1 for x in models_results[name1]], [x+1 for x in models_results[name2]],
                            name1, name2, models_colors[name1], models_colors[name2], labels, out_file_path, logscale, n)
        else:
            create_ranks2D_plot(models_results[name1], models_results[name2],
                            name1, name2, models_colors[name1], models_colors[name2], labels, out_file_path, logscale, n)

def create_ranks2D_plot(x_axis, y_axis, name_x, name_y, color_x, color_y, labels, out_file_path=None, logscale=False, n=None):
    colors = []
    for x, y in zip(x_axis, y_axis):
        if y > x:
            colors.append(color_x)
        elif y < x:
            colors.append(color_y)
        else:
            colors.append('dimgrey')

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

    # fig.canvas.mpl_connect("motion_notify_event", hover)

    # Plot the diagonal line
    plt.plot([1, 10000], [1, 10000], color='gray', linestyle='--', linewidth=0.5)

    # plt.title(f"{name_x}   vs   {name_y}")
    # plt.xlabel(f"'{name_x}' Rank {'(logscale)' if logscale else ''}")
    # plt.ylabel(f"'{name_y}' Rank {'(logscale)' if logscale else ''}")

    if logscale:
        axx = plt.gca()
        axx.set_xscale('log')
        axx.set_yscale('log')

    # plt.xlim(0,10)
    # plt.ylim(0,10)

    legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_x, markersize=15, label=f'{name_x} ({colors.count(color_x)})'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_y, markersize=15, label=f'{name_y} ({colors.count(color_y)})'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='dimgrey', markersize=15, label=f'{"equally good"} ({colors.count("dimgrey")})')
    ]

    # Add legend to plot
    plt.legend(handles=legend_elements,fontsize='13')

    # Save if path specified
    if out_file_path is not None:
        plt.savefig(out_file_path, dpi=600, bbox_inches='tight', pad_inches=0)

    # Show the plot
    plt.show()

def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode images using a specified model.")
    parser.add_argument("--dataset",default="lsc", type=str, help="Dataset name", choices=["marine", "photos", "marine_compet", "lsc"])
    parser.add_argument("--labels",default="long", type=str, help="Type of labels", choices=["short", "long"])
    parser.add_argument("--tight_layout", default=True, type=bool, help="True if no title and axis labels should be displayed")
    parser.add_argument("--interactive", default=False, type=bool, help="True if labels should show if you point at them")
    parser.add_argument("--models", default=['openclip-ViT_SO400M_14_webli','openclip-ViT_L_16_webli','openclip-ViT_B_16_webli','openclip-ViT_H_14_dfn5b','openclip-ViT_G_14_laion2b' ],
                        type=list, help="Selected models ['clip', 'blip2', 'align', 'openclip-ViT_SO400M_14_webli','openclip-ViT_L_16_webli','openclip-ViT_B_16_webli','openclip-ViT_H_14_dfn5b','openclip-ViT_G_14_laion2b','openclip-ViT_H_14_laion2b' ]")
    args = parser.parse_args()


    
    
    df_labels = pd.read_csv(f'datasets/{args.dataset}/labels/labels.csv')
    labels_list = sorted(df_labels[f'{args.labels}_label'].to_list())

    

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
    
        
    plot_ranks2D(d,colors_d,labels_list, out_dir=f"saves/plots/plot2D/{args.dataset}_{args.labels}/")