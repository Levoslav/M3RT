import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

def save_images_in_grid(image_names, output_pdf):
    # Convert 5 pixels to inches (assuming 72 pixels per inch)
    pixel_spacing = 5 / 72

    # Create a figure with 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 9.5))  # A4 size in inches: 8.27 x 11.69
    axes = axes.flatten()

    # Loop through the list of image names and add each image to the grid
    for ax, img_name in zip(axes, image_names):
        img = mpimg.imread(img_name)
        ax.imshow(img)
        ax.axis('off')  # Hide axes

    # Remove any empty subplots
    for ax in axes[len(image_names):]:
        fig.delaxes(ax)

    # Adjust the spacing between the images
    plt.subplots_adjust(left=pixel_spacing, right=1-pixel_spacing, 
                        top=1-pixel_spacing, bottom=pixel_spacing, 
                        wspace=pixel_spacing, hspace=pixel_spacing)

    # Add text labels
    # fig.text(0.01, 0.81, 'Private Photos Dataset', va='center', rotation='vertical', fontsize=12)
    # fig.text(0.01, 0.51, 'MVK', va='center', rotation='vertical', fontsize=12)
    # fig.text(0.01, 0.21, 'LSC', va='center', rotation='vertical', fontsize=12)
    
    fig.text(0.30, 0.95, 'Short Text Labels', ha='center', fontsize=12)
    fig.text(0.75, 0.95, 'Long Text Labels', ha='center', fontsize=12)

    # Save the figure as a high-quality PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig, bbox_inches='tight', dpi=600)  # Increase dpi for better quality

# mvk
# path = 'datasets/marine/data/'
# image_names = ['00088.jpg', '06754.jpg', '00839.jpg', '02326.jpg', '07163.jpg', '06137.jpg']
# image_names = [path+name for name in image_names]
# output_pdf = 'mvk.pdf'

# private
# path = 'datasets/photos/data/'
# image_names = ['14.jpeg','103.jpeg','17.jpeg','305.jpeg','506.jpeg','428.jpeg']
# image_names = [path+name for name in image_names]
# output_pdf = "private_photos.pdf"

path = 'saves/plots/cumulative_graphs/'
image_names = ['photos_short.png','photos_long.png','marine_short.png','marine_long.png','lsc_short.png','lsc_long.png']
image_names = [path+name for name in image_names]
output_pdf = "cumulative_graphs.pdf"

save_images_in_grid(image_names, output_pdf)
