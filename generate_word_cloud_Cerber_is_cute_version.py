import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIG ---
OUTPUT_NAME = "chat_wordcloud.png"
MASK_IMAGE_PATH = "inputs/Minawan color Drawing large.png"
FONT_PATH = "C:/Users/rober/AppData/Local/Microsoft/Windows/Fonts/RobotoSlab-VariableFont_wght.ttf"

# Load mask
mask_img = Image.open(MASK_IMAGE_PATH).convert("RGB")
MASK_IMAGE = np.array(mask_img)
h, w = MASK_IMAGE.shape[:2]

# Create boolean mask: True = allowed for placement
# Here: non-white pixels are allowed
mask_allowed = np.any(MASK_IMAGE < 250, axis=2)

# Words and counts
words_to_draw = {"GooseChanWan": 1000, "honk": 1}
font_size_range = (20, 50)
honk_size = 10

# Occupancy map to prevent overlaps
occupied = np.zeros((h, w), dtype=bool)

def can_place(x, y, w_word, h_word):
    if x + w_word >= w or y + h_word >= h:
        return False
    if occupied[y:y+h_word, x:x+w_word].any():
        return False
    if not mask_allowed[y:y+h_word, x:x+w_word].all():
        return False
    return True

def mark_occupied(x, y, w_word, h_word):
    occupied[y:y+h_word, x:x+w_word] = True

# Create image
img = Image.new("RGB", (w, h), color="black")
draw = ImageDraw.Draw(img)

for word, count in words_to_draw.items():
    for i in range(count):
        font_size = honk_size if word == "honk" else random.randint(*font_size_range)
        font = ImageFont.truetype(FONT_PATH, font_size)
        bbox = font.getbbox(word)
        word_width = bbox[2] - bbox[0]
        word_height = bbox[3] - bbox[1]

        # Try random positions until we find a valid one
        for attempt in range(500):
            x = random.randint(0, w - word_width)
            y = random.randint(0, h - word_height)
            if can_place(x, y, word_width, word_height):
                # Sample color from mask
                fill_color = tuple(MASK_IMAGE[y, x][:3])
                draw.text((x, y), word, font=font, fill=fill_color)
                mark_occupied(x, y, word_width, word_height)
                break
        else:
            print(f"⚠️ Could not place word '{word}' (attempted 500 times)")

# --- Save output ---
os.makedirs("outputs", exist_ok=True)
output_path = os.path.join("outputs", OUTPUT_NAME)
img.save(output_path)
print(f"✅ Saved word cloud as {output_path}")

# Optional: display
plt.imshow(img)
plt.axis("off")
plt.show()
