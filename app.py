"""
Skin Shade Matcher — Streamlit App
Run: streamlit run shade_matcher_app.py

Install dependencies first:
    pip install streamlit opencv-python scikit-learn pandas numpy Pillow
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import io
import os

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Shade Match",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #FFFFFF; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 600;
    color: #000000 !important;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-size: 1.1rem;
    color: #222222 !important;
    font-weight: 400;
    margin-bottom: 2rem;
}

.skin-profile-card {
    background: #F4F4F4;
    border: 1px solid #CCCCCC;
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
}

.profile-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #000000 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.profile-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #000000 !important;
    font-weight: 600;
}

.match-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    border: 1px solid #E5E5E5;
    transition: box-shadow 0.2s;
    display: flex;
    align-items: center;
    gap: 1rem;
}

.match-card:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.10);
}

.rank-badge {
    width: 32px;
    height: 32px;
    background: #111111;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 600;
    flex-shrink: 0;
}

.shade-swatch {
    width: 48px;
    height: 48px;
    border-radius: 10px;
    border: 2px solid rgba(0,0,0,0.08);
    flex-shrink: 0;
}

.brand-name {
    font-size: 0.72rem;
    font-weight: 500;
    color: #555555;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}

.shade-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    color: #111111;
    font-weight: 600;
}

.delta-badge {
    background: #F0F0F0;
    color: #333333;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 4px;
    display: inline-block;
}

.undertone-badge {
    font-size: 0.72rem;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    display: inline-block;
    margin-left: 4px;
}

.undertone-Warm   { background: #FFF0D0; color: #7A4500; }
.undertone-Cool   { background: #D8EAFF; color: #003580; }
.undertone-Neutral{ background: #EBEBEB; color: #333333; }

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #111111;
    margin: 1.5rem 0 0.8rem;
}

.stButton > button {
    background: #111111 !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover { opacity: 0.80 !important; }

.divider {
    height: 1px;
    background: #E5E5E5;
    margin: 1.5rem 0;
}

.no-match {
    background: #F8F8F8;
    border: 1px dashed #CCCCCC;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #444444;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def rgb_to_lab(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    def lin(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    r, g, b = lin(r), lin(g), lin(b)
    X = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) / 0.95047
    Y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) / 1.00000
    Z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) / 1.08883
    def f(t):
        return t ** (1/3) if t > 0.008856 else 7.787 * t + 16/116
    fx, fy, fz = f(X), f(Y), f(Z)
    return round(116 * fy - 16, 4), round(500 * (fx - fy), 4), round(200 * (fy - fz), 4)

def get_undertone(a, b):
    if b > 18:            return "Warm"
    elif b < 10 or a > 12: return "Cool"
    else:                  return "Neutral"

def get_skin_tone(ita):
    if ita > 55:   return "Very Light"
    elif ita > 41: return "Light"
    elif ita > 28: return "Intermediate"
    elif ita > 10: return "Tan"
    elif ita > -30: return "Brown"
    else:           return "Dark"

def hex_to_rgb_tuple(hex_str):
    h = str(hex_str).strip().lstrip('#')
    if len(h) != 6: return (200, 170, 150)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def extract_skin_from_pil(pil_image):
    """Extract average skin Lab from a PIL image using face detection."""
    img_array = np.array(pil_image.convert('RGB'))
    img_bgr   = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        return None, "No face detected. Please use a clear, front-facing photo."

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    regions = {
        "forehead":    img_bgr[y+int(h*.05):y+int(h*.22), x+int(w*.30):x+int(w*.70)],
        "left_cheek":  img_bgr[y+int(h*.45):y+int(h*.65), x+int(w*.10):x+int(w*.35)],
        "right_cheek": img_bgr[y+int(h*.45):y+int(h*.65), x+int(w*.65):x+int(w*.90)],
    }

    all_pixels = []
    for region in regions.values():
        if region.size == 0: continue
        pixels     = region.reshape(-1, 3).astype(float)
        brightness = pixels.mean(axis=1)
        pixels     = pixels[(brightness > 40) & (brightness < 220)]
        if len(pixels): all_pixels.append(pixels)

    if not all_pixels:
        return None, "Could not sample skin pixels. Try better lighting."

    combined          = np.vstack(all_pixels)
    avg_b, avg_g, avg_r = combined.mean(axis=0)
    L, A, B           = rgb_to_lab(avg_r, avg_g, avg_b)

    # Draw face box on image for preview
    preview = img_array.copy()
    cv2.rectangle(preview, (x, y), (x+w, y+h), (220, 120, 80), 3)
    for name, region_coords in [
        ("forehead",   (x+int(w*.30), y+int(h*.05), x+int(w*.70), y+int(h*.22))),
        ("left_cheek", (x+int(w*.10), y+int(h*.45), x+int(w*.35), y+int(h*.65))),
        ("right_cheek",(x+int(w*.65), y+int(h*.45), x+int(w*.90), y+int(h*.65))),
    ]:
        cv2.rectangle(preview,
                      (region_coords[0], region_coords[1]),
                      (region_coords[2], region_coords[3]),
                      (80, 180, 120), 2)

    return {
        "L": L, "A": A, "B": B,
        "avg_rgb": (int(avg_r), int(avg_g), int(avg_b)),
        "face_box": (x, y, w, h),
        "preview_img": Image.fromarray(preview)
    }, None


@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)
    df.dropna(subset=['L', 'A', 'B_lab'], inplace=True)
    return df

def build_knn(df):
    knn = NearestNeighbors(n_neighbors=min(10, len(df)), metric='euclidean')
    knn.fit(df[['L', 'A', 'B_lab']].values)
    return knn

def find_matches(knn, df, user_L, user_A, user_B, top_n, filter_undertone):
    distances, indices = knn.kneighbors([[user_L, user_A, user_B]])
    matches = df.iloc[indices[0]].copy()
    matches['delta_E'] = distances[0].round(4)
    if filter_undertone != "All":
        matches = matches[matches['undertone'] == filter_undertone]
    return matches.head(top_n)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    dataset_path = st.text_input(
        "Dataset path",
        value=r"C:\Users\arvin\OneDrive\Desktop\shade match\shades.csv",
        help="Full path to your shades.csv file"
    )

    top_n = st.slider("Number of matches", 3, 10, 5)

    filter_undertone = st.selectbox(
        "Filter by undertone",
        ["All", "Warm", "Cool", "Neutral"]
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📖 About")
    st.markdown("""
    <small style='color:#444444'>
    This app analyses your skin tone from a selfie or manual Lab input
    and finds the closest matching foundation & concealer shades
    from the Kaggle cosmetics dataset using KNN in Lab color space.
    </small>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════
st.markdown("<div class='hero-title' style='color:#000000!important'>💄 Shade Match</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-sub' style='color:#222222!important'>Find your perfect foundation &amp; concealer — powered by skin science</div>",
            unsafe_allow_html=True)

# Load dataset
if not os.path.exists(dataset_path):
    st.error(f"Dataset not found at: `{dataset_path}`\n\nPlease update the path in the sidebar.")
    st.stop()

df  = load_dataset(dataset_path)
knn = build_knn(df)
st.success(f"Dataset loaded — {len(df)} shades across **{df['brand'].nunique()}** brands")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ── Input Mode Tabs ───────────────────────────────────────────
tab1, tab2 = st.tabs(["📸  Upload Selfie", "🎨  Manual Input"])

user_lab   = None
avg_rgb    = None
preview_img = None

# ── TAB 1: Selfie ─────────────────────────────────────────────
with tab1:
    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("#### Upload your photo")
        st.caption("Use a well-lit, front-facing photo for best results. Avoid heavy filters.")
        uploaded = st.file_uploader(
            "Choose image", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="Uploaded photo", use_column_width=True)

            if st.button("🔍 Analyse Skin Tone", key="analyse_btn"):
                with st.spinner("Detecting face and sampling skin pixels..."):
                    result, err = extract_skin_from_pil(pil_img)
                if err:
                    st.error(err)
                else:
                    st.session_state['user_lab'] = (result['L'], result['A'], result['B'])
                    st.session_state['avg_rgb']  = result['avg_rgb']
                    st.session_state['preview']  = result['preview_img']
                    st.success("Skin tone extracted!")

    with col_preview:
        if 'preview' in st.session_state:
            st.markdown("#### Detection preview")
            st.image(st.session_state['preview'],
                     caption="Green = sampled regions", use_column_width=True)

    if 'user_lab' in st.session_state:
        user_lab = st.session_state['user_lab']
        avg_rgb  = st.session_state.get('avg_rgb')

# ── TAB 2: Manual ─────────────────────────────────────────────
with tab2:
    st.markdown("#### Enter your skin Lab values")
    st.caption("You can find these from a color picker tool or from a previous analysis.")

    col_l, col_a, col_b = st.columns(3)
    with col_l:
        man_L = st.number_input("L* (Lightness)", 0.0, 100.0, 70.0, 0.1,
                                 help="0 = black, 100 = white")
    with col_a:
        man_A = st.number_input("a* (Red-Green)", -50.0, 50.0, 10.0, 0.1,
                                 help="Positive = reddish, Negative = greenish")
    with col_b:
        man_B = st.number_input("b* (Yellow-Blue)", -50.0, 50.0, 20.0, 0.1,
                                 help="Positive = yellow/warm, Negative = blue/cool")

    st.markdown("##### Or pick a skin color")
    skin_color = st.color_picker("Skin color picker", "#D4956A")
    hr, hg, hb = hex_to_rgb_tuple(skin_color)

    col_use1, col_use2 = st.columns(2)
    with col_use1:
        if st.button("Use Lab values above", key="use_lab"):
            st.session_state['user_lab'] = (man_L, man_A, man_B)
            st.session_state['avg_rgb']  = None
            st.session_state.pop('preview', None)
            st.success("Lab values set!")
    with col_use2:
        if st.button("Use color picker", key="use_picker"):
            L, A, B = rgb_to_lab(hr, hg, hb)
            st.session_state['user_lab'] = (L, A, B)
            st.session_state['avg_rgb']  = (hr, hg, hb)
            st.session_state.pop('preview', None)
            st.success(f"Color → Lab: L={L:.1f}, A={A:.1f}, B={B:.1f}")

    if 'user_lab' in st.session_state:
        user_lab = st.session_state['user_lab']
        avg_rgb  = st.session_state.get('avg_rgb')


# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

if user_lab:
    uL, uA, uB = user_lab
    ita         = float(np.degrees(np.arctan2(uL - 50, uB)))
    skin_tone   = get_skin_tone(ita)
    undertone   = get_undertone(uA, uB)
    avg_r, avg_g, avg_b = avg_rgb if avg_rgb else (200, 160, 130)

    # ── Skin Profile ─────────────────────────────────────────
    st.markdown("<div class='section-title'>Your Skin Profile</div>", unsafe_allow_html=True)

    col_swatch, col_p1, col_p2, col_p3, col_p4 = st.columns([1, 2, 2, 2, 2])
    with col_swatch:
        st.markdown(
            f"<div style='width:70px;height:70px;border-radius:14px;"
            f"background:rgb({avg_r},{avg_g},{avg_b});"
            f"border:2px solid rgba(0,0,0,0.1);margin-top:8px'></div>",
            unsafe_allow_html=True
        )
    with col_p1:
        st.markdown(f"<div class='profile-label' style='color:#000000;font-weight:700'>Skin Tone</div>"
                    f"<div class='profile-value' style='color:#000000'>{skin_tone}</div>", unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"<div class='profile-label' style='color:#000000;font-weight:700'>Undertone</div>"
                    f"<div class='profile-value' style='color:#000000'>{undertone}</div>", unsafe_allow_html=True)
    with col_p3:
        st.markdown(f"<div class='profile-label' style='color:#000000;font-weight:700'>ITA Angle</div>"
                    f"<div class='profile-value' style='color:#000000'>{ita:.1f}°</div>", unsafe_allow_html=True)
    with col_p4:
        st.markdown(f"<div class='profile-label' style='color:#000000;font-weight:700'>Lab Values</div>"
                    f"<div class='profile-value' style='font-size:1rem;color:#000000'>"
                    f"L={uL:.1f} A={uA:.1f} B={uB:.1f}</div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Matches ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>Your Shade Matches</div>", unsafe_allow_html=True)

    matches = find_matches(knn, df, uL, uA, uB, top_n, filter_undertone)

    if matches.empty:
        st.markdown("<div class='no-match'>No matches found for the selected undertone filter. "
                    "Try setting undertone filter to <b>All</b>.</div>", unsafe_allow_html=True)
    else:
        for i, (_, row) in enumerate(matches.iterrows(), 1):
            r2, g2, b2   = hex_to_rgb_tuple(row.get('hex', 'D4956A'))
            ut_class     = f"undertone-{row.get('undertone','Neutral')}"
            delta        = row.get('delta_E', 0)
            delta_label  = "✦ Best match" if delta < 2 else f"ΔE {delta:.2f}"
            hex_val      = str(row.get('hex', '')).lstrip('#')

            st.markdown(f"""
            <div class='match-card'>
                <div class='rank-badge'>{i}</div>
                <div class='shade-swatch' style='background:#{hex_val}'></div>
                <div style='flex:1'>
                    <div class='brand-name'>{row.get('brand','')}</div>
                    <div class='shade-name'>{row.get('product_short','')}</div>
                    <span class='delta-badge'>{delta_label}</span>
                    <span class='undertone-badge {ut_class}'>{row.get('undertone','')}</span>
                </div>
                <div style='text-align:right;min-width:80px'>
                    <div style='font-size:0.72rem;color:#555555'>Hex</div>
                    <div style='font-size:0.9rem;color:#111111;font-weight:500'>#{hex_val}</div>
                    <div style='font-size:0.72rem;color:#555555;margin-top:4px'>
                        L={row.get('L',0):.1f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Download ─────────────────────────────────────────────
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    if not matches.empty:
        csv_data = matches.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️  Download my matches as CSV",
            data=csv_data,
            file_name="my_shade_matches.csv",
            mime="text/csv"
        )

else:
    st.markdown("""
    <div class='no-match'>
        <div style='font-size:2rem;margin-bottom:0.5rem'>💄</div>
        <div style='font-family:Playfair Display,serif;font-size:1.2rem;color:#111111;margin-bottom:0.5rem'>
            Ready to find your match
        </div>
        Upload a selfie or enter Lab values above to get started.
    </div>
    """, unsafe_allow_html=True)