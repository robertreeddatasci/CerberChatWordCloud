import pandas as pd
import re
import glob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
INPUT_FOLDER = "Twitch Chat"  # folder containing your CSVs
OUTPUT_ALL = "chat_wordcloud.png"
OUTPUT_MEANINGFUL = "meaningful_chat_wordcloud.png"

# --- EMOTES ---
emotes =   ['cerbyRave2',
            'cerbyMinaLove',
            'cerbyMinaJam',
            'cerbySpin',
            'cerbyMILC',
            'cerbyBounce',
            ':)',
            'cerbyMinaWave',
            '<3',
            'cerbyHug',
            'cerbyMinaGun',
            'cerbyLUL',
            'cerbyDinkDonk',
            'cerbyBlushi',
            'cerbyBlush',
            'cerbyLoad',
            'cerbyRave1',
            'cerbyMinaArrive',
            ';)',
            'HeyGuys',
            'Kappa',
            'LUL',
            'PogChamp',
            'VoHiYo',
            'cerbyWaggy',
            'cerbyBlushi',
            'cerbyHype',
            'cerbyVeryAngy',
            'cerbyLoad',
            'cerbyRave1',
            'cerbyRave2',
            'cerbyFine1',
            'cerbyMunch',
            'cerbyGOODWAN',
            'cerbyPetpet',
            'cerbyBounce',
            'cerbySpin',
            'cerbyTail',
            'cerbyZoomies',
            'cerbyGremlin',
            'cerbySoCute',
            'cerbyWAN',
            'cerbyEmby',
            'cerbyMinaArrive',
            'cerbyMinaJam',
            'cerbyMinaBounce',
            'cerbyMinaExcited',
            'cerbyMinaGun',
            'cerbyMinaDoko',
            'cerbyMinaWave',
            'cerbyMinaSmug',
            'cerbyMinaBlush',
            'cerbyMinaPat',
            'cerbyDinkDonk',
            'cerbyMinaLick',
            'cerbyGURU',
            'cerbyPat',
            'cerbyStinky',
            'cerbyPant',
            'cerbyFlower',
            'cerbySad',
            'cerbyHeart',
            'cerbyWow',
            'cerbyWave',
            'cerbySmug',
            'cerbyPeek',
            'cerbyLUL',
            'cerbyLove',
            'cerbyPet',
            'cerbyCry',
            'cerbyYAY',
            'cerbyBlush',
            'cerbyLurk',
            'cerbyMILC',
            'cerbyHMPF',
            'cerbyLoading',
            'cerbyDead',
            'cerbyCozy',
            'cerbyAngy',
            'cerbyHuh',
            'cerbyRave',
            'cerbyDum',
            'cerbyPopcorn',
            'cerbyFine',
            'cerbyUno',
            'cerbyMILM',
            'cerbyMILS',
            'cerbyWanwan',
            'cerbyMILO',
            'cerbyCerbwaa',
            'cerbyCool',
            'cerbyMILE',
            'cerbyHide',
            'cerbyPLSNO',
            'cerbyWoah',
            'cerbyWeep',
            'cerbyStabbyKnifey',
            'cerbyPoint',
            'cerbyOwO',
            'cerbyWHEYYY',
            'cerbyHug',
            'cerbyMinaLove',
            'cerbyCerby']


# --- HELPER FUNCTION: Filter messages ---
def is_meaningful(text):
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 15:  # too short
        return False
    words = text.split()
    if len(set(words)) == 1:  # repeated spam
        return False
    if re.fullmatch(r"(\W+|\w+){1,3}", text):  # 1-3 tokens only
        return False
    return True

# --- STEP 1: Load all CSV files ---
all_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
print(all_files)
if not all_files:
    raise ValueError(f"No CSV files found in {INPUT_FOLDER}")

dfs = []
for f in all_files:
    df = pd.read_csv(f, quotechar='"', engine="python", usecols=[0,1,2,3])

    df.columns = [c.lower() for c in df.columns]  # normalize column names
    df["source_file"] = os.path.basename(f)      # keep track of file
    dfs.append(df)

print(len(dfs))

# # Combine all CSVs into one DataFrame
# combined_df = pd.concat(dfs, ignore_index=True)
# messages = combined_df["message"].dropna().tolist()
# print(f"Loaded {len(messages)} messages from {len(all_files)} CSV files.")

# # --- STEP 2: Define stopwords ---
# stopwords = set(STOPWORDS)
# stopwords.update(["lol", "lmao", "pog", "gg", "haha"])  # common chat noise

# # --- STEP 3: Generate WordCloud for ALL messages ---
# print("Generating full chat word cloud...")
# all_text = " ".join(messages)
# wordcloud_all = WordCloud(
#     width=4000, height=2000,
#     background_color="white",
#     stopwords=stopwords,
#     collocations=False,
#     min_font_size=10
# ).generate(all_text)

# plt.figure(figsize=(40, 20), dpi=300)
# plt.imshow(wordcloud_all, interpolation="bilinear")
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.savefig(OUTPUT_ALL, dpi=300)
# plt.close()

# # --- STEP 4: Filter meaningful messages ---
# meaningful = combined_df[combined_df["message"].apply(is_meaningful)]["message"].tolist()
# print(f"Generating meaningful chat word cloud with {len(meaningful)} messages...")

# meaningful_text = " ".join(meaningful)
# wordcloud_meaningful = WordCloud(
#     width=4000, height=2000,
#     background_color="white",
#     stopwords=stopwords,
#     collocations=False,
#     min_font_size=10
# ).generate(meaningful_text)

# plt.figure(figsize=(40, 20), dpi=300)
# plt.imshow(wordcloud_meaningful, interpolation="bilinear")
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.savefig(OUTPUT_MEANINGFUL, dpi=300)
# plt.close()

# print(f"✅ Word clouds saved as {OUTPUT_ALL} and {OUTPUT_MEANINGFUL}")



