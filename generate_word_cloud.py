import pandas as pd
import re
import glob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

# --- CONFIGURATION ---
INPUT_FOLDER = "Twitch Chat"
OUTPUT_ALL = "chat_wordcloud.png"
OUTPUT_MEANINGFUL = "meaningful_chat_wordcloud.png"
MASK_IMAGE = np.array(Image.open("Minawan Drawing large.png"))

# --- CUSTOM STOPWORDS ---
stopwords = set(STOPWORDS)
stopwords.update(["lol", "lmao", "pog", "gg", "haha", "tier", "sub"])

# --- PHRASES TO BLOCK ---
BLOCKED_PHRASES = [
    "is gifting",
    "gifted a total of",
    "just subscribed",
    "just resubscribed"
]

# --- FUNCTION: Filter Messages ---
def is_meaningful(message: str) -> bool:
    msg = message.strip()

    for phrase in BLOCKED_PHRASES:
        if phrase.lower() in msg.lower():
            return False

    if msg.startswith("!"):  # commands
        return False

    if len(msg) < 55:  # too short
        return False

    if not re.search(r'[A-Za-z]', msg):  # no letters
        return False

    words = re.findall(r'\w+', msg.lower())
    unique_words = set(words)

    if len(unique_words) / max(len(words), 1) < 0.3:  # heavy repetition
        return False

    return True


# --- FUNCTION: Load and Parse CSVs ---
def load_chat_data(folder: str) -> pd.DataFrame:
    records = []
    file_counts = {}

    for f in glob.glob(os.path.join(folder, "*.csv")):
        line_count = 0
        with open(f, "r", encoding="utf-8") as infile:
            header = infile.readline().strip()
            for line in infile:
                parts = line.rstrip("\n").split(",", 3)
                if len(parts) == 4:
                    time, user_name, user_color, message = parts
                    if message.startswith('"') and message.endswith('"'):
                        message = message[1:-1]
                    records.append([time, user_name, user_color, message])
                    line_count += 1
                else:
                    print(f"⚠️ Skipped malformed line in {f}: {line[:80]}...")
        file_counts[os.path.basename(f)] = line_count

    df = pd.DataFrame(records, columns=["time", "user_name", "user_color", "message"])
    print("✅ File loading complete. Lines per file:")
    for file, count in file_counts.items():
        print(f"  {file}: {count} lines")
    print(f"TOTAL: {len(df)} messages across {len(file_counts)} files.")
    return df


# --- FUNCTION: Generate Word Cloud ---
def generate_wordcloud(text: str, output_file: str, mask_image=None, use_image_colors=True):
    print(f"Generating word cloud → {output_file}")

    wordcloud = WordCloud(
        width=6000,
        height=3000,
        background_color="white",
        mask=mask_image,
        stopwords=stopwords,
        contour_width=3,
        contour_color="black",
        collocations=False,
        min_font_size=6,
        max_font_size=100,
        max_words=5000
    ).generate(text)

    if use_image_colors and mask_image is not None:
        image_colors = ImageColorGenerator(mask_image)
        wordcloud.recolor(color_func=image_colors)

    plt.figure(figsize=(40, 20), dpi=300)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"✅ Saved word cloud as {output_file}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    df = load_chat_data(INPUT_FOLDER)

    # --- ALL MESSAGES ---
    all_text = " ".join(df["message"].dropna().astype(str).tolist())
    generate_wordcloud(all_text, OUTPUT_ALL, mask_image=MASK_IMAGE)

    # --- MEANINGFUL MESSAGES ---
    df["is_meaningful"] = df["message"].apply(is_meaningful)
    meaningful_df = df[df["is_meaningful"]]
    meaningful_df.to_csv("Meaningful_df.csv", index=False)

    meaningful_text = " ".join(meaningful_df["message"].dropna().astype(str).tolist())
    generate_wordcloud(meaningful_text, OUTPUT_MEANINGFUL, mask_image=MASK_IMAGE)