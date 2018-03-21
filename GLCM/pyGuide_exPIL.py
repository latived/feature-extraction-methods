# Mudei: diretório, format em .save, removi _getexif porque .tiff não possue
from PIL import Image, ImageFilter
# Read image
im = Image.open("/home/lativ/Documents/UFAL/GioconDa/"
    "Dados/ImagesDataset/ColonDB/CVC-ColonDB/CVC-ColonDB/1.tiff"
    ) # o interpreteador juntará as linhas até os parênteses fecharem
# Display image
im.show()

# Applying a filter to the image
im_sharp = im.filter( ImageFilter.SHARPEN )
# Saving the filtered image to a new file
im_sharp.save('1_image_sharpened.jpg')

# Spitting the image into its respective bands, i.e. Red, Green, and Blue for RGB
r, g, b = im_sharp.split()

