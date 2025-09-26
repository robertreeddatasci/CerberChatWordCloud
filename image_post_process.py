from PIL import Image

def apply_image_mask(wordcloud_path, color_image_path, output_path):
    """
    Takes a word cloud image (white text on black background) and a color image,
    and fills the text with the colors from the image like a mask.

    Parameters:
        wordcloud_path (str): Path to the word cloud image.
        color_image_path (str): Path to the color/shape image (e.g. turtle).
        output_path (str): Path to save the final composited image.
    """
    # Load images
    wc_img = Image.open(wordcloud_path).convert("L")   # Grayscale: letters = bright, background = dark
    color_img = Image.open(color_image_path).convert("RGB")

    # Resize color image to match word cloud size if needed
    if color_img.size != wc_img.size:
        color_img = color_img.resize(wc_img.size, Image.LANCZOS)

    # Create a binary mask from word cloud
    # Threshold: turn light areas (letters) to white (mask = 1)
    mask = wc_img.point(lambda p: 255 if p > 10 else 0).convert("1")

    # Composite: put color image wherever mask is white, black elsewhere
    final_img = Image.composite(color_img, Image.new("RGB", color_img.size, "black"), mask)

    # Save result
    final_img.save(output_path)
    print(f"✅ Masked image saved to: {output_path}")

# Example usage:
apply_image_mask("usernames_wordcloud.png", "turtle.png", "turtle_wordcloud_colored.png")
