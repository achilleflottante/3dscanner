from PIL import Image


amask = Image.open(r"D:\Ecole\Cours\info\scanner\cropped.png")
image = Image.open(r"depthanythingai/media/boite.PNG")
bg = Image.new("RGBA", image.size, (215,215,215,255))

amask.paste(image, mask=amask)

amask.save("croppeddepth.png")

bg.paste(amask, mask=amask.split()[3])
bg.save("depthfinal.png")
amask = Image.open(r"D:\Ecole\Cours\info\scanner\cropped.png")
bg = Image.new("RGBA", image.size, (215,215,215,255))

bg.paste(amask, mask=amask.split()[3])
bg.save("colorfinal.png")