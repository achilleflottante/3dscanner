from rembg import remove
from PIL import Image
import numpy as np

image = Image.open(r"D:\Ecole\Cours\info\scanner\depthmaps\1d.PNG")

image = np.array(image)
image = image /1.5 + 70
image[0, 0] = 0

im = image.astype(int)
print(im)

image = Image.fromarray(image)
im = Image.fromarray(im)


im.convert('L')
im.show()
im.save("lighter.png")
