from typing import Tuple
from PIL import Image
import numpy as np
import cv2, copy

class ColorHelper():

    def is_color_similar(color1, color2, threshold=5):
        return abs(color1[0] - color2[0]) < threshold and abs(color1[1] - color2[1]) < threshold and abs(color1[2] - color2[2]) < threshold

    @staticmethod
    def isPixelLineBlank(x1, x2, y1, y2, image, regions, base):
        for x in range(x1, x2):
            for y in range(y1, y2):
                if not ColorHelper.is_color_similar(image.getpixel((x, y)), (255, 255, 255)): 
                    return False
                pixel = (x, y)
                for region in regions:
                    if region == base.xyxy:
                        continue
                    if x >= region[0] and x <= region[2] and y >= region[1] and y <= region[3]:
                        return False
        return True

class Rectangle():

    def __init__(self, area) -> None:
        self.area = area
        self.x = area[0]
        self.y = area[1]
        self.x2 = area[2]
        self.y2 = area[3]
        self.xyxy = [self.x, self.y, self.x2, self.y2]

    def visualiseOnImage(self, image: Image):
        size = image.size
        color = (255, 0, 0)

        for x in range(self.x, self.x2):
            image.putpixel((x, self.y), color)
            image.putpixel((x, self.y2), color)
        
        for y in range(self.y, self.y2):
            image.putpixel((self.x, y), color)
            image.putpixel((self.x2, y), color)


    @staticmethod
    def expand_rectangle(image: Image, rectangle, regions):
        size = image.size
        color = image.getpixel((rectangle.x, rectangle.y))
        base = copy.copy(rectangle)

        while ColorHelper.isPixelLineBlank(rectangle.x, rectangle.x2, rectangle.y - 1, rectangle.y, image, regions, base):
            rectangle.y -= 1
            
        while ColorHelper.isPixelLineBlank(rectangle.x, rectangle.x2, rectangle.y2, rectangle.y2 + 1, image, regions, base):
            rectangle.y2 += 1

        while ColorHelper.isPixelLineBlank(rectangle.x - 1, rectangle.x, rectangle.y, rectangle.y2, image, regions, base):
            rectangle.x -= 1
            
        while ColorHelper.isPixelLineBlank(rectangle.x2, rectangle.x2 + 1, rectangle.y, rectangle.y2, image, regions, base):
            rectangle.x2 += 1

        rectangle.area = (rectangle.x, rectangle.y, rectangle.x2, rectangle.y2)
                
        return rectangle

if __name__ == "__main__":
    img = Image.open("inpaint_input.png")

    all = [([763, 297, 948, 446], "I'LL USE THESE COUNTERFEIT MERCHANT UNION BILLS"), ([77, 441, 262, 590], 'TO IGNITE THE FUSE FOR A COLLAPSE OF TRUST...!!!'), ([796, 943, 935, 1044], 'I TOOK OVER THE MERCHANT UNION,'), ([334, 873, 510, 1018], "ONCE I'M DONE TRANSFERRING THE CULT'S FUNDS TO A SAFE LOCATION, I WILL BEGIN THE OPERATION."), ([651, 1068, 815, 1237], 'AND INTENTIONALLY GAVE THEIR BILLS A CRUDE DESIGN, ALL FOR THE SAKE OF THIS MOMENT.'), ([70, 1272, 215, 1407], 'NO MATTER HOW MUCH COMBAT STRENGTH MITSUGOSHI HAS...')]
    # all = [([763, 297, 948, 446], "I'LL USE THESE COUNTERFEIT MERCHANT UNION BILLS"), ([796, 943, 935, 1044], 'I TOOK OVER THE MERCHANT UNION,'), ([651, 1068, 815, 1237], 'AND INTENTIONALLY GAVE THEIR BILLS A CRUDE DESIGN, ALL FOR THE SAKE OF THIS MOMENT.')]
    # all = [([763, 297, 948, 446], "I'LL USE THESE COUNTERFEIT MERCHANT UNION BILLS")]
    regions = [i[0] for i in all]
    new = []
    for area in regions:
        a = Rectangle(area)
        a = Rectangle.expand_rectangle(img, a, regions)
        new.append(a)
    for i in new:
        i.visualiseOnImage(img)
    img.show()