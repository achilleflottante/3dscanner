from rembg import remove
from PIL import Image

image = Image.open(r"D:\Ecole\Cours\info\scanner\depthmaps\2d.PNG")

image = remove(image)

png_info = image.info
image.save('croppedblack.png', **png_info)