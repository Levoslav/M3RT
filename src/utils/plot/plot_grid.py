import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
import numpy as np
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages

def plot_images_in_grid(N, image_paths,names, output_pdf_path):
    # Load images
    images = [Image.open(path).resize((3193, 2388), Image.ANTIALIAS) for path in image_paths]
    
    # Determine the size of the mask image based on the first image
    width, height = images[4].size
    print(width, height)
    print(images[0].size)
    

    # Calculate figure size 
    fig_width = 8.5  
    fig_height = fig_width * (height / width) 
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = ImageGrid(fig, 111, nrows_ncols=(N, N), axes_pad=0.1, cbar_mode=None)

    img_idx = 0
    for i in range(N):
        for j in range(N):
            ax = grid[i * N + j]
            if i == j:
                ax.imshow(Image.open("mask_image.png").resize((3193, 2388), Image.ANTIALIAS))
            else:
                ax.imshow(images[img_idx])
                img_idx += 1

            ax.axis('off')

            # Add titles to the top row images
            if i == 0 and names:
                ax.set_title(names[j], fontsize=6, pad=10)

    # if column_names:
    #     for i in range(N):
    #         fig.text(0.01, (N - i - 0.5) / N, column_names[i], va='center', rotation='vertical',ha='right', fontsize=10)
    # fig.text(0.01, 0.01 , column_names[i], va='center', rotation='vertical',ha='right', fontsize=10,alpha=0.5)
    # fig.text(0.01, 0.99 , column_names[i], va='center', rotation='vertical',ha='right', fontsize=10)

    # plt.subplots_adjust(left=0.001)
    

    # Save the figure to a PDF
    with PdfPages(output_pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight',dpi=600)
        plt.close(fig)



dataset = 'lsc'
labels = 'long'
N = 5  # Grid size
upper = f'saves/plots/plot2D/{dataset}_{labels}/'
lower = f'saves/plots/cumulative_complementary_graphs/{dataset}_{labels}/'
names = ['ViT_SO400M_14_webli','ViT_L_16_webli','ViT_B_16_webli','ViT_H_14_dfn5b','ViT_G_14_laion2b' ]
image_paths = []
for i,r in enumerate(names):
    for j,c in enumerate(names):
        file_name = f"{r}-{c}.png"
        if i == j:
            pass
            # diagonal do nothing
        elif i < j:
            image_paths.append(upper+file_name)
        elif j < i:
            image_paths.append(lower+file_name)


output_pdf = f'{dataset}_{labels}.pdf'

plot_images_in_grid(N,image_paths,names,output_pdf)