"""
Diagnostic: sends ONLY an image to the VLM, with no surrounding text
context, to isolate whether the model can actually read it in isolation.

Run:  python diagnose_image.py --image data\images\page513_render.png
"""
import argparse
import base64
import ollama

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--model", default="llava:13b")
args = parser.parse_args()

with open(args.image, "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

prompt = (
    "Look carefully at this image of a fuse panel diagram. List every "
    "fuse label and its amperage rating that you can read in the grid. "
    "Be exhaustive - go box by box."
)

response = ollama.chat(
    model=args.model,
    messages=[
        {
            "role": "user",
            "content": prompt,
            "images": [image_data],
        }
    ],
)

print(response["message"]["content"])
