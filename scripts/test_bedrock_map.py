import json
import sys

from bedrock_detector import detect_walls_with_bedrock


with open(sys.argv[1], "rb") as image_file:
    image = image_file.read()

width = int(sys.argv[2])
height = int(sys.argv[3])
print(json.dumps(detect_walls_with_bedrock(image, width, height), indent=2))
