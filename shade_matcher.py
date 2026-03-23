import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import cv2
import os

# ── Paths ─────────────────────────────────────────────────────
SHADES_CSV   = r"C:\Users\arvin\OneDrive\Desktop\shade match\shades_full.csv"
SKIN_IMG_DIR = r"C:\Users\arvin\OneDrive\Desktop\shade match\train"  # your skin tone image dataset

# ── Load dataset ──────────────────────────────────────────────
df = pd.read_csv(SHADES_CSV)
print(f"Loaded {len(df)} shades | Brands: {df['brand'].unique()}")

# ── Build KNN model on Lab features ──────────────────────────
features = df[['L', 'A', 'B_lab']].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(features)

knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(X_scaled)
print("KNN model ready.")

# ════════════════════════════════════════════════════════════
# HELPER: RGB → Lab (pure numpy, no colormath)
# ════════════════════════════════════════════════════════════
def rgb_to_lab_single(r, g, b):
    arr = np.array([r, g, b], dtype=float) / 255.0
    arr = np.where(arr > 0.04045,
                   ((arr + 0.055) / 1.055) ** 2.4,
                   arr / 12.92)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = M @ arr
    xyz /= [0.95047, 1.00000, 1.08883]
    def f(t):
        return t**(1/3) if t > (6/29)**3 else t/(3*(6/29)**2) + 4/29
    fx, fy, fz = f(xyz[0]), f(xyz[1]), f(xyz[2])
    L     = 116 * fy - 16
    A_val = 500 * (fx - fy)
    B_val = 200 * (fy - fz)
    return round(L, 4), round(A_val, 4), round(B_val, 4)

def undertone_from_lab(a_val, b_val):
    if b_val > 18:   return "Warm"
    elif b_val < 10 or a_val > 12: return "Cool"
    else:            return "Neutral"

def ita_category(ita):
    if   ita > 55:  return "Very Light"
    elif ita > 41:  return "Light"
    elif ita > 28:  return "Intermediate"
    elif ita > 10:  return "Tan"
    elif ita > -30: return "Brown"
    else:           return "Dark"

# ════════════════════════════════════════════════════════════
# CORE: Extract skin tone from image
# ════════════════════════════════════════════════════════════
def extract_skin_lab_from_image(image_path):
    """
    Load an image, detect face region, sample skin pixels
    from forehead + cheeks, return average Lab values.
    Works for both selfies and dataset skin tone images.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w    = img_rgb.shape[:2]

    # Try face detection first
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    if len(faces) > 0:
        # Use the largest detected face
        x, y, fw, fh = max(faces, key=lambda f: f[2]*f[3])

        # Sample 3 ROIs: forehead, left cheek, right cheek
        rois = [
            img_rgb[y + int(fh*0.05) : y + int(fh*0.25),
                    x + int(fw*0.25) : x + int(fw*0.75)],   # forehead
            img_rgb[y + int(fh*0.45) : y + int(fh*0.70),
                    x + int(fw*0.05) : x + int(fw*0.30)],   # left cheek
            img_rgb[y + int(fh*0.45) : y + int(fh*0.70),
                    x + int(fw*0.70) : x + int(fw*0.95)],   # right cheek
        ]
        pixels = np.vstack([roi.reshape(-1, 3) for roi in rois if roi.size > 0])
    else:
        # No face found — use center crop (works for flat skin tone dataset images)
        print(f"  No face detected in {os.path.basename(image_path)} — using center crop")
        cx, cy = w // 2, h // 2
        crop   = img_rgb[cy-40:cy+40, cx-40:cx+40]
        pixels = crop.reshape(-1, 3)

    # Filter out near-black and near-white pixels (artifacts)
    brightness = pixels.mean(axis=1)
    pixels     = pixels[(brightness > 40) & (brightness < 230)]

    if len(pixels) == 0:
        raise ValueError("No usable skin pixels found in image.")

    avg_r, avg_g, avg_b = pixels.mean(axis=0)
    return rgb_to_lab_single(avg_r, avg_g, avg_b), (avg_r, avg_g, avg_b)

# ════════════════════════════════════════════════════════════
# CORE: Match skin Lab to products
# ════════════════════════════════════════════════════════════
def match_shades(L, A, B_lab, top_n=5, filter_undertone=True):
    """
    Given user's skin Lab values, return top_n matching products.
    Optionally filter by undertone for better recommendations.
    """
    user_vec   = scaler.transform([[L, A, B_lab]])
    distances, indices = knn.kneighbors(user_vec, n_neighbors=min(20, len(df)))

    results = df.iloc[indices[0]].copy()
    results['distance'] = distances[0]

    user_undertone = undertone_from_lab(A, B_lab)
    user_ita       = np.degrees(np.arctan2(L - 50, B_lab))

    if filter_undertone:
        filtered = results[results['undertone'] == user_undertone]
        results  = filtered if len(filtered) >= top_n else results

    return results.head(top_n), user_undertone, ita_category(user_ita)

# ════════════════════════════════════════════════════════════
# MAIN: Run on a single image
# ════════════════════════════════════════════════════════════
def analyze_image(image_path):
    print(f"\n{'='*55}")
    print(f"Analyzing: {os.path.basename(image_path)}")
    print(f"{'='*55}")

    (L, A, B_lab), (r, g, b) = extract_skin_lab_from_image(image_path)

    print(f"Extracted skin color → RGB: ({r:.0f}, {g:.0f}, {b:.0f})")
    print(f"Lab values          → L={L:.2f}, A={A:.2f}, B={B_lab:.2f}")

    matches, undertone, skin_tone = match_shades(L, A, B_lab, top_n=5)

    print(f"\nUser skin tone  : {skin_tone}")
    print(f"User undertone  : {undertone}")
    print(f"\nTop 5 recommended shades:")
    print("-" * 55)

    display_cols = ['brand', 'product_short', 'hex',
                    'skin_tone', 'undertone', 'distance']
    print(matches[display_cols].to_string(index=False))
    print("-" * 55)

    return matches

# ════════════════════════════════════════════════════════════
# BATCH: Run on entire skin tone image dataset folder
# ════════════════════════════════════════════════════════════
def batch_analyze_folder(folder_path, max_images=10):
    """
    Process multiple skin tone images from your downloaded dataset.
    Saves results to a CSV.
    """
    supported = ('.jpg', '.jpeg', '.png', '.webp')
    images    = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(supported)][:max_images]

    if not images:
        print(f"No images found in {folder_path}")
        return

    print(f"\nProcessing {len(images)} images from {folder_path}...")
    all_results = []

    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        try:
            (L, A, B_lab), (r, g, b) = extract_skin_lab_from_image(img_path)
            matches, undertone, skin_tone = match_shades(L, A, B_lab, top_n=3)

            for _, row in matches.iterrows():
                all_results.append({
                    'image_file' : img_file,
                    'skin_R'     : round(r),
                    'skin_G'     : round(g),
                    'skin_B'     : round(b),
                    'skin_L'     : L,
                    'skin_A'     : A,
                    'skin_B_lab' : B_lab,
                    'skin_tone'  : skin_tone,
                    'undertone'  : undertone,
                    'brand'      : row['brand'],
                    'product'    : row['product_short'],
                    'shade_hex'  : row['hex'],
                    'distance'   : round(row['distance'], 4),
                })
            print(f"  ✓ {img_file} → {skin_tone} | {undertone}")
        except Exception as e:
            print(f"  ✗ {img_file} → Error: {e}")

    out_df = pd.DataFrame(all_results)
    out_path = r"C:\Users\arvin\OneDrive\Desktop\shade match\batch_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nBatch results saved → {out_path}")
    return out_df

# ════════════════════════════════════════════════════════════
# MANUAL INPUT: type in shade + undertone directly
# ════════════════════════════════════════════════════════════
def match_manual(hex_color=None, r=None, g=None, b=None):
    """
    Match by manually providing a hex color or RGB values.
    Example: match_manual(hex_color='e2aa7b')
             match_manual(r=200, g=160, b=120)
    """
    if hex_color:
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

    L, A, B_lab = rgb_to_lab_single(r, g, b)
    print(f"\nManual input → RGB({r},{g},{b})")
    print(f"Lab → L={L:.2f}, A={A:.2f}, B={B_lab:.2f}")

    matches, undertone, skin_tone = match_shades(L, A, B_lab, top_n=5)
    print(f"Skin tone: {skin_tone} | Undertone: {undertone}")
    print("\nTop matches:")
    print(matches[['brand','product_short','hex','undertone','distance']].to_string(index=False))
    return matches

# ════════════════════════════════════════════════════════════
# RUN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # Option 1: Test with one image from your skin tone dataset
    sample_images = [f for f in os.listdir(SKIN_IMG_DIR)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))]

    if sample_images:
        test_img = os.path.join(SKIN_IMG_DIR, sample_images[0])
        analyze_image(test_img)
    else:
        print("No images found in train folder.")

    # Option 2: Batch process your skin tone dataset
    # batch_analyze_folder(SKIN_IMG_DIR, max_images=20)

    # Option 3: Manual hex input (uncomment to use)
    # match_manual(hex_color='e2aa7b')
    # match_manual(r=200, g=160, b=120)