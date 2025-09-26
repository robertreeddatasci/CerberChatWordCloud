import pandas as pd
import numpy as np
import os
import re
import glob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime

# --- CONFIGURATION ---
INPUT_FOLDER = "Cerber_Twitch_Chat"
OUTPUT_ALL = "chat_wordcloud.png"
OUTPUT_MEANINGFUL = "meaningful_chat_wordcloud.png"
OUTPUT_USERNAMES = "usernames_wordcloud.png"
MASK_IMAGE = np.array(Image.open("inputs/Minawan color Drawing large.png")) #"Minawan color purple Drawing large.png"
h, w = MASK_IMAGE.shape[:2] 
print(f'h:{h}, w:{w}')

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

def get_counts(df):
    print(df.head())

    user_message_counts_df = (
        df.groupby("user_name")
        .agg(message_count=("message", "count"))
        .reset_index()
        .sort_values("message_count", ascending=False)
        .reset_index(drop=True)
    )

    # Compute percentages
    total_messages = user_message_counts_df["message_count"].sum()
    user_message_counts_df["percentage"] = (
        user_message_counts_df["message_count"] / total_messages * 100
    ).round(2)

    print(user_message_counts_df.head(25))
    return


# --- FUNCTION: Generate Word Cloud ---
def generate_wordcloud(text: str, output_file: str, mask_image=None, use_image_colors=True, timestamp=True):
    """
    Generate a word cloud from the given text and save it inside an 'outputs' subfolder.

    Parameters:
        text (str): The text to generate the word cloud from.
        output_file (str): The filename to use for saving the image (e.g., 'wordcloud.png').
        mask_image (ndarray or None): Mask to shape the word cloud (optional).
        use_image_colors (bool): Whether to recolor the words based on the mask image.
        timestamp (bool): If True, adds a timestamp to the filename to avoid overwriting.
    """
    # --- Ensure outputs folder exists ---
    os.makedirs("outputs", exist_ok=True)

    # --- Optionally add timestamp to filename ---
    if timestamp:
        name, ext = os.path.splitext(output_file)
        output_file = f"{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{ext}"

    output_path = os.path.join("outputs", output_file)

    print(f"Generating word cloud → {output_path}")

    # --- Generate the word cloud ---
    wordcloud = WordCloud(
        width=w,
        height=h,
        background_color="black",
        mask=mask_image,
        stopwords=stopwords,
        contour_width=3,
        contour_color="white",
        collocations=False,
        min_font_size=12,
        max_font_size=70,
        max_words=10000,
        font_path="C:/Users/rober/AppData/Local/Microsoft/Windows/Fonts/RobotoSlab-VariableFont_wght.ttf"
    ).generate(text)

    # --- Recolor based on mask image if requested ---
    if use_image_colors and mask_image is not None:
        image_colors = ImageColorGenerator(mask_image)
        wordcloud.recolor(color_func=image_colors)

    # --- Save at high resolution with no padding ---
    wc_array = wordcloud.to_array()
    h_px, w_px = wc_array.shape[:2]
    dpi = 300
    plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    plt.imshow(wc_array, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"✅ Saved word cloud as {output_path}")
    return output_path

def generate_wordcloud_highres(
    text: str,
    output_file: str,
    mask_image=None,
    use_image_colors=True,
    timestamp=True,
    scale_factor=2
):
    """
    Generate a high-resolution word cloud optimized for zooming, modeled after the original function.

    Parameters:
        text (str): The text to generate the word cloud from.
        output_file (str): The filename to use for saving the image (e.g., 'wordcloud.png').
        mask_image (ndarray or None): Mask to shape the word cloud (optional).
        use_image_colors (bool): Whether to recolor the words based on the mask image.
        timestamp (bool): If True, adds a timestamp to the filename to avoid overwriting.
        scale_factor (int): Factor to scale width and height for higher resolution.
    """
    # --- Ensure outputs folder exists ---
    os.makedirs("outputs", exist_ok=True)

    # --- Optionally add timestamp to filename ---
    if timestamp:
        name, ext = os.path.splitext(output_file)
        output_file = f"{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{ext}"

    output_path = os.path.join("outputs", output_file)

    print(f"Generating high-res word cloud → {output_path}")

    # --- Determine scaled width and height ---
    if mask_image is not None:
        h_scaled, w_scaled = mask_image.shape[:2]
        h_scaled *= scale_factor
        w_scaled *= scale_factor
    else:
        w_scaled, h_scaled = 2400*scale_factor, 1600*scale_factor  # fallback size

    # --- Generate the word cloud ---
    wordcloud = WordCloud(
        width=w_scaled,
        height=h_scaled,
        background_color="black",
        mask=mask_image,
        stopwords=stopwords,
        contour_width=3,
        contour_color="white",
        collocations=False,
        min_font_size=12,
        max_font_size=70,
        max_words=10000,
        font_path="C:/Users/rober/AppData/Local/Microsoft/Windows/Fonts/RobotoSlab-VariableFont_wght.ttf"
    ).generate(text)

    # --- Recolor based on mask image if requested ---
    if use_image_colors and mask_image is not None:
        image_colors = ImageColorGenerator(mask_image)
        wordcloud.recolor(color_func=image_colors)

    # --- Save at high resolution with no padding ---
    wc_array = wordcloud.to_array()
    h_px, w_px = wc_array.shape[:2]
    dpi = 300
    plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    plt.imshow(wc_array, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"✅ Saved high-res word cloud as {output_path}")
    return output_path




# --- MAIN EXECUTION ---
if __name__ == "__main__":
    df = load_chat_data(INPUT_FOLDER)

    # --- ALL MESSAGES ---
    all_text = " ".join(df["message"].dropna().astype(str).tolist())
    generate_wordcloud_highres(all_text, OUTPUT_ALL, mask_image=MASK_IMAGE)

    # --- MEANINGFUL MESSAGES ---
    df["is_meaningful"] = df["message"].apply(is_meaningful)
    meaningful_df = df[df["is_meaningful"]]
    meaningful_df.to_csv("outputs/Meaningful_df.csv", index=False)

    meaningful_text = " ".join(meaningful_df["message"].dropna().astype(str).tolist())
    generate_wordcloud_highres(meaningful_text, OUTPUT_MEANINGFUL, mask_image=MASK_IMAGE)

    # Keep only usernames with ASCII alphanumeric characters
    alphanumeric_usernames = df["user_name"].dropna().astype(str).apply(
        lambda x: x if re.fullmatch(r'[A-Za-z0-9]+', x) else None
    ).dropna()

    # Join them for the word cloud
    usernames_text = " ".join(alphanumeric_usernames.tolist())
    generate_wordcloud_highres(usernames_text, OUTPUT_USERNAMES, mask_image=MASK_IMAGE)