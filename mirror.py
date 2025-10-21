mirror_map = {
    'A': 'A', 'B': 'ꓭ', 'C': 'Ɔ', 'D': 'ᗡ', 'E': 'Ǝ',
    'F': 'ꟻ', 'G': '⅁', 'H': 'H', 'I': 'I', 'J': '⅃',
    'K': 'ꓘ', 'L': '⅃', 'M': 'M', 'N': 'N', 'O': 'O',
    'P': 'Ԁ', 'Q': 'Ꝗ', 'R': 'ꓤ', 'S': 'S', 'T': 'T',
    'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z',
    ',': ',', '.': '.', '!': '!', '?': '?', ' ': ' '
}

original_string = "HELLO, WORLD!"
mirrored = "".join(mirror_map.get(ch, ch) for ch in original_string[::-1])
print(mirrored)