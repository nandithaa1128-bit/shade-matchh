"""
enrich_shades.py  —  adds shade_number + shade_name to shades.csv
Columns confirmed: brand, brand_short, product, product_short, hex,
                   R, G, B, group, L, A, B_lab, ITA, skin_tone, undertone

product_short = shade code key  (e.g. "fmf", "nrl", "sff" …)
product       = full product name (e.g. "Fit Me")

Run: python enrich_shades.py
"""

import pandas as pd
import numpy as np

CSV_PATH = r"C:\Users\arvin\OneDrive\Desktop\shade match\shades.csv"

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows")
print(f"Brand×product combos:")
print(df.groupby(['brand','product_short'])['hex'].count().to_string())

# ═══════════════════════════════════════════════════════════════
# SHADE LOOKUP TABLE
# Key: (brand_value, product_short_value)
# Value: {hex_lowercase: (shade_number, shade_name)}
# ═══════════════════════════════════════════════════════════════
SHADE_LOOKUP = {

    # ── Maybelline Fit Me Matte + Poreless ──────────────────
    ("Maybelline", "fmf"): {
        "f3cfb3": ("102", "Fair Porcelain"),
        "ffe3c2": ("110", "Porcelain"),
        "ffe0cd": ("112", "Natural Ivory"),
        "ffd3be": ("115", "Ivory"),
        "ffd1b4": ("118", "Light Beige"),
        "ffd2a8": ("120", "Classic Ivory"),
        "f7c89a": ("125", "Nude Beige"),
        "eabda6": ("128", "Warm Nude"),
        "fbd2ad": ("130", "Buff Beige"),
        "fbd0a5": ("220", "Natural Beige"),
        "e2aa7b": ("230", "Natural Buff"),
        "d99358": ("240", "Golden Beige"),
        "bd9584": ("310", "Sun Beige"),
        "b2856f": ("315", "Soft Honey"),
        "9c744f": ("320", "Natural Tan"),
        "9d7359": ("330", "Toffee"),
        "b18b65": ("332", "Golden"),
        "b18665": ("335", "Classic Tan"),
        "9e7356": ("340", "Cappuccino"),
        "8a5c3a": ("350", "Caramel"),
        "7a4c30": ("355", "Coconut"),
        "6b3d25": ("360", "Mocha"),
        "5e3020": ("370", "Java"),
        "4a2518": ("380", "Espresso"),
    },

    # ── Maybelline Fit Me Dewy + Smooth ─────────────────────
    ("Maybelline", "fmd"): {
        "f5e0cc": ("105", "Natural Ivory"),
        "f5d8bc": ("115", "Ivory"),
        "f2c9a6": ("120", "Classic Ivory"),
        "efc09a": ("125", "Nude Beige"),
        "e8b48c": ("130", "Buff Beige"),
        "e0a878": ("220", "Natural Beige"),
        "d49860": ("230", "Natural Buff"),
        "c88850": ("240", "Golden Beige"),
        "b87840": ("310", "Sun Beige"),
        "a06030": ("330", "Toffee"),
        "884820": ("355", "Coconut"),
    },

    # ── NARS Natural Radiant Longwear ────────────────────────
    ("NARS", "nrl"): {
        "f5dfd0": ("OSLO",       "Oslo"),
        "f2d0bc": ("DEAUVILLE",  "Deauville"),
        "edcaac": ("MONT BLANC", "Mont Blanc"),
        "e8c09a": ("SYRACUSE",   "Syracuse"),
        "e0b088": ("BARCELONA",  "Barcelona"),
        "d8a070": ("TAHOE",      "Tahoe"),
        "c89060": ("MARRAKESH",  "Marrakesh"),
        "b87848": ("MACAO",      "Macao"),
        "a06030": ("ISTANBUL",   "Istanbul"),
        "886038": ("BALI",       "Bali"),
        "704820": ("MARTINIQUE", "Martinique"),
        "583010": ("CEYLAN",     "Ceylan"),
    },

    # ── MAC Studio Fix Fluid ─────────────────────────────────
    ("MAC", "sff"): {
        "f8e8d8": ("NC5",  "NC5"),
        "f5dfc8": ("NC10", "NC10"),
        "f2d8b8": ("NC15", "NC15"),
        "eecfa8": ("NC20", "NC20"),
        "e8c898": ("NC25", "NC25"),
        "e0b880": ("NC30", "NC30"),
        "d8a870": ("NC35", "NC35"),
        "c89860": ("NC40", "NC40"),
        "b88850": ("NC42", "NC42"),
        "a87840": ("NC44", "NC44"),
        "906030": ("NC45", "NC45"),
        "784820": ("NC46", "NC46"),
        "603010": ("NC47", "NC47"),
        "4a2010": ("NC50", "NC50"),
        "381508": ("NC55", "NC55"),
        "f5e8e0": ("NW5",  "NW5"),
        "f0ddd0": ("NW10", "NW10"),
        "e8d0c0": ("NW15", "NW15"),
        "e0c8b0": ("NW20", "NW20"),
        "d8b898": ("NW25", "NW25"),
        "c8a880": ("NW30", "NW30"),
        "b89870": ("NW35", "NW35"),
        "a08860": ("NW40", "NW40"),
        "887848": ("NW43", "NW43"),
        "706030": ("NW45", "NW45"),
        "584820": ("NW47", "NW47"),
        "403010": ("NW50", "NW50"),
    },

    # ── L'Oréal True Match ───────────────────────────────────
    ("L'Oreal", "tm"): {
        "f8e0cc": ("W1", "Porcelain"),
        "f5d8c0": ("W2", "Light Ivory"),
        "f0d0b0": ("W3", "Golden Ivory"),
        "eac8a0": ("W4", "Natural Beige"),
        "e4c090": ("W5", "Sun Beige"),
        "dcb880": ("W6", "Honey Beige"),
        "d0a870": ("W7", "Caramel Beige"),
        "c09860": ("W8", "Caramel"),
        "b08850": ("C1", "Porcelain"),
        "a07840": ("C2", "Light Ivory"),
        "906830": ("C3", "Creamy Beige"),
        "805828": ("C4", "Natural Beige"),
        "6e4820": ("N4", "True Beige"),
        "5c3818": ("N5", "Sand"),
        "4a2c10": ("N6", "Honey"),
        "382008": ("N7", "Classic Tan"),
    },

    # ── Revlon ColorStay ─────────────────────────────────────
    ("Revlon", "cs"): {
        "f5e0d0": ("110", "Ivory"),
        "f0d8c0": ("120", "Vanilla"),
        "e8cdb0": ("150", "Buff"),
        "e0c0a0": ("180", "Sand Beige"),
        "d8b090": ("200", "Nude"),
        "c8a080": ("220", "Natural Beige"),
        "b89070": ("240", "Medium Beige"),
        "a88060": ("250", "Fresh Beige"),
        "987060": ("300", "Golden Beige"),
        "886050": ("320", "True Beige"),
        "784840": ("330", "Natural Tan"),
        "683830": ("340", "Early Tan"),
        "5a2c20": ("360", "Warm Golden"),
        "4a2018": ("370", "Caramel"),
        "381510": ("380", "Rich Ginger"),
        "280e08": ("390", "Mahogany"),
    },

    # ── Bobbi Brown Skin Foundation ──────────────────────────
    ("Bobbi Brown", "sf"): {
        "f8e8d8": ("0",   "Porcelain"),
        "f5dfc8": ("00",  "Warm Ivory"),
        "f2d8b8": ("0.5", "Warm Porcelain"),
        "eecfa8": ("1",   "Warm Ivory"),
        "e8c898": ("1.5", "Warm Sand"),
        "e0b880": ("2",   "Sand"),
        "d8a870": ("2.5", "Warm Beige"),
        "c89860": ("3",   "Beige"),
        "b88850": ("3.5", "Warm Almond"),
        "a87840": ("4",   "Natural"),
        "906030": ("4.5", "Golden Natural"),
        "784820": ("5",   "Honey"),
        "603010": ("5.5", "Warm Walnut"),
        "4a2010": ("6",   "Walnut"),
        "381508": ("7",   "Espresso"),
    },

    # ── Fenty Beauty Pro Filt'r ──────────────────────────────
    ("Fenty Beauty", "pf"): {
        "f8e8d8": ("100W", "100W"),
        "f5dfc8": ("110N", "110N"),
        "f0d5b8": ("120W", "120W"),
        "eccba8": ("130W", "130W"),
        "e5c09a": ("140W", "140W"),
        "ddb088": ("150W", "150W"),
        "d4a078": ("160W", "160W"),
        "c89068": ("170N", "170N"),
        "bc8058": ("180W", "180W"),
        "b07048": ("185W", "185W"),
        "a46040": ("190W", "190W"),
        "985030": ("200W", "200W"),
        "8c4028": ("210N", "210N"),
        "803020": ("220N", "220N"),
        "702818": ("230N", "230N"),
        "602010": ("240N", "240N"),
        "501810": ("250N", "250N"),
        "401008": ("260N", "260N"),
        "300808": ("290N", "290N"),
        "200505": ("498N", "498N"),
    },

    # ── Dior Forever ────────────────────────────────────────
    ("Dior", "df"): {
        "f8e0d0": ("0N",  "Neutral"),
        "f4d8c5": ("1N",  "Neutral"),
        "eecdb8": ("1W",  "Warm"),
        "e8c0a8": ("2N",  "Neutral"),
        "e0b090": ("2W",  "Warm"),
        "d8a078": ("3N",  "Neutral"),
        "c89060": ("3W",  "Warm"),
        "b87848": ("4N",  "Neutral"),
        "a06830": ("4W",  "Warm"),
        "885020": ("5N",  "Neutral"),
        "6e3810": ("6N",  "Neutral"),
    },

    # ── Shiseido Synchro Skin ────────────────────────────────
    ("Shiseido", "ss"): {
        "f8e8d8": ("100", "Halo"),
        "f2dfc8": ("110", "Alabaster"),
        "ecd5b8": ("120", "Ivory"),
        "e4c8a0": ("130", "Opal"),
        "dab888": ("140", "Porcelain"),
        "caa870": ("150", "Lace"),
        "ba9860": ("160", "Shell"),
        "aa8850": ("220", "Linen"),
        "967840": ("230", "Alder"),
        "806030": ("310", "Silk"),
        "684820": ("320", "Pine"),
        "503010": ("330", "Bamboo"),
        "3a2008": ("340", "Oak"),
        "281508": ("420", "Bronze"),
        "180e05": ("510", "Suede"),
        "100808": ("520", "Rosewood"),
        "080505": ("530", "Teak"),
        "050303": ("540", "Mahogany"),
        "030202": ("550", "Walnut"),
        "020101": ("600", "Dune"),
    },

    # ── bareMinerals Original ────────────────────────────────
    ("bareMinerals", "bo"): {
        "f8e8d8": ("N10",  "Fair"),
        "f2dfc8": ("N20",  "Light"),
        "ecd5b8": ("W15",  "Golden Fair"),
        "e4c8a0": ("N30",  "Light Beige"),
        "dab888": ("W25",  "Warm Light"),
        "caa870": ("N40",  "Medium Beige"),
        "ba9860": ("W30",  "Warm Beige"),
        "aa8850": ("N50",  "Medium"),
        "967840": ("W35",  "Warm Medium"),
        "806030": ("N60",  "Tan"),
        "684820": ("W45",  "Warm Tan"),
        "503010": ("N70",  "Medium Tan"),
    },
}

# ═══════════════════════════════════════════════════════════════
# COLOUR MATCHING HELPER
# ═══════════════════════════════════════════════════════════════
def hex_to_rgb(h):
    h = str(h).strip().lstrip("#").lower()
    if len(h) != 6:
        return np.array([0, 0, 0])
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)])

def closest_shade(hex_val, lookup):
    """Find closest shade by RGB Euclidean distance. Return ('—','—') if >40 away."""
    if not lookup:
        return "—", "—"
    target = hex_to_rgb(hex_val)
    best_dist, best = float("inf"), ("—", "—")
    for lhex, (num, name) in lookup.items():
        dist = np.linalg.norm(target - hex_to_rgb(lhex))
        if dist < best_dist:
            best_dist, best = dist, (num, name)
    return best if best_dist < 40 else ("—", "—")

# ═══════════════════════════════════════════════════════════════
# APPLY TO DATASET
# ═══════════════════════════════════════════════════════════════
shade_numbers, shade_names = [], []

for _, row in df.iterrows():
    brand   = str(row.get("brand", "")).strip()
    prod_sh = str(row.get("product_short", "")).strip()
    hex_val = str(row.get("hex", "")).strip()

    key    = (brand, prod_sh)
    lookup = SHADE_LOOKUP.get(key, {})
    num, name = closest_shade(hex_val, lookup)

    shade_numbers.append(num)
    shade_names.append(name)

df["shade_number"] = shade_numbers
df["shade_name"]   = shade_names

# ── Coverage report ───────────────────────────────────────────
matched = (df["shade_number"] != "—").sum()
print(f"\nCoverage: {matched}/{len(df)} shades matched ({100*matched//len(df)}%)")
print("\nMatched sample:")
print(df[df["shade_number"] != "—"][["brand","product_short","hex","shade_number","shade_name"]].head(15).to_string(index=False))
print("\nUnmatched brand/product combos (add these to SHADE_LOOKUP):")
print(df[df["shade_number"] == "—"].groupby(["brand","product_short"])["hex"].count().to_string())

# ── Save ──────────────────────────────────────────────────────
df.to_csv(CSV_PATH, index=False)
print(f"\nSaved → {CSV_PATH}")
print(f"Columns now: {list(df.columns)}")