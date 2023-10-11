from PIL import Image, ImageFilter

def create_turing_pattern(path, rep=20, radius=2, sharpen_percent=300, size=None):
    img = Image.open(path).convert('LA')
    if size is not None:
        img.thumbnail(size, Image.ANTIALIAS)
    for _ in range(rep):
        img = img.filter(ImageFilter.BoxBlur(radius=radius))
        img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=sharpen_percent, threshold=0))
    return img.convert("RGB")


img = create_turing_pattern("glitch.test.png")
img.save("test.glitch.jpg")
