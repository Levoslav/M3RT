import matplotlib.pyplot as plt
# import numpy as np
from itertools import combinations
import os
import pickle
import pandas as pd

def plot_ranks2D(models_results: dict, labels, n=50):
    names = models_results.keys()
    for name1, name2 in list(combinations(names, 2)):
        create_ranks2D_plot(models_results[name1], models_results[name2],
                            name1, name2, labels, n)

def create_ranks2D_plot(x_axis, y_axis, name_x, name_y, labels, n=50):
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
    plt.plot([0, n], [0, n], color='gray', linestyle='--', linewidth=0.5)

    plt.title(f"{name_x} vs {name_y}")
    plt.xlabel(f"{name_x} rank")
    plt.ylabel(f"{name_y} rank")
    plt.xlim(0, n)
    plt.ylim(0, n)

    # Show the plot
    plt.show()

def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

clip_ranks = load_from_file('CLIP_ranks.pkl')
align_ranks = load_from_file('ALIGN_ranks.pkl')
blip2_ranks = load_from_file('BLIP2_ranks.pkl')
openclip_ranks = load_from_file('OpenCLIP_ranks.pkl')
clip_ranks = [b for a,b in clip_ranks]
align_ranks = [b for a,b in align_ranks]
blip2_ranks = [b for a,b in blip2_ranks]
openclip_ranks = [b for a,b in openclip_ranks]



d = {
    'CLIP':clip_ranks,
    'ALIGN':align_ranks,
    'BLIP2':blip2_ranks,
    'OpenCLIP':openclip_ranks
}
df_labels = pd.read_csv('labels.csv')
labels = sorted(list(df_labels.label))

plot_ranks2D(d,labels)