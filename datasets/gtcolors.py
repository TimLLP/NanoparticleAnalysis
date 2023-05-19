from PIL import Image
# ctrl + alt +l
image = Image.open("/home/swu/peng/mycode/datasets/image1399.png")
img = image.convert('RGBA')
total = []
for i in range(256):
    for j in range(256):
        r, g, b, a = img.getpixel((i, j))
    if r not in total:
        total.append([r])

print(total)
