"""
Shade Match — FastAPI Backend v3
Run: uvicorn main:app --reload
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import cv2, numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
import io, os
from PIL import Image

app = FastAPI(title="Shade Match API v3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

# ═══════════════════════════════════════════════════════════════
# LIPSTICK TABLE
# ═══════════════════════════════════════════════════════════════
LIPSTICK_TABLE = {
    ("Very Light","Warm"):[
        {"category":"Nude","brand":"MAC","shade":"Velvet Teddy","hex":"B5836A","desc":"Warm greige nude","shop_url":"https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Velvet_Teddy","shop_platform":"Brand Site"},
        {"category":"Nude","brand":"NYX","shade":"Creme Brulee","hex":"D4956A","desc":"Peachy beige"},
        {"category":"Coral","brand":"L'Oréal","shade":"Coral Bistro","hex":"E8734A","desc":"Warm coral"},
        {"category":"Coral","brand":"Maybelline","shade":"Peach Pop","hex":"F0926A","desc":"Soft peach"},
        {"category":"Bold","brand":"MAC","shade":"Russian Red","hex":"9E1B32","desc":"Classic warm red"},
        {"category":"Bold","brand":"NYX","shade":"Alabama","hex":"8B1A2E","desc":"Deep brick red"},
    ],
    ("Very Light","Cool"):[
        {"category":"Nude","brand":"MAC","shade":"Kinda Sexy","hex":"C4927A","desc":"Pink-nude"},
        {"category":"Nude","brand":"NYX","shade":"Babe","hex":"D4A0A0","desc":"Sheer pink"},
        {"category":"Coral","brand":"L'Oréal","shade":"Tickled Pink","hex":"E87A8A","desc":"Cool coral-pink"},
        {"category":"Coral","brand":"Revlon","shade":"Pink Velvet","hex":"E8909A","desc":"Rosy pink"},
        {"category":"Bold","brand":"MAC","shade":"Rebel","hex":"6B2D4E","desc":"Deep berry"},
        {"category":"Bold","brand":"NYX","shade":"Bordeaux","hex":"5C1A2E","desc":"Wine berry"},
    ],
    ("Very Light","Neutral"):[
        {"category":"Nude","brand":"MAC","shade":"Velvet Teddy","hex":"B5836A","desc":"Warm greige nude","shop_url":"https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Velvet_Teddy","shop_platform":"Brand Site"},
        {"category":"Nude","brand":"Maybelline","shade":"Barely There","hex":"C8A088","desc":"Natural nude"},
        {"category":"Coral","brand":"L'Oréal","shade":"Sunset Coral","hex":"E8835A","desc":"True coral"},
        {"category":"Coral","brand":"Revlon","shade":"Coral Berry","hex":"E07060","desc":"Coral pink"},
        {"category":"Bold","brand":"MAC","shade":"Ruby Woo","hex":"9B1B30","desc":"Iconic true red"},
        {"category":"Bold","brand":"NYX","shade":"Black Cherry","hex":"4A1228","desc":"Dark berry"},
    ],
    ("Light","Warm"):[
        {"category":"Nude","brand":"MAC","shade":"Honey Love","hex":"C48A60","desc":"Golden nude"},
        {"category":"Nude","brand":"NYX","shade":"Praline","hex":"B87850","desc":"Warm caramel"},
        {"category":"Coral","brand":"L'Oréal","shade":"Coral Clair","hex":"E8784A","desc":"Bright warm coral"},
        {"category":"Coral","brand":"Maybelline","shade":"Coral Spice","hex":"D4623A","desc":"Terracotta coral"},
        {"category":"Bold","brand":"MAC","shade":"Lady Danger","hex":"C8302A","desc":"Bright warm red"},
        {"category":"Bold","brand":"NYX","shade":"Dark Era","hex":"6E1A10","desc":"Deep auburn red"},
    ],
    ("Light","Cool"):[
        {"category":"Nude","brand":"MAC","shade":"Twig","hex":"B88878","desc":"Muted rose-nude"},
        {"category":"Nude","brand":"NYX","shade":"London","hex":"C09080","desc":"Cool mauve nude"},
        {"category":"Coral","brand":"Revlon","shade":"Rose Velvet","hex":"D4707A","desc":"Cool rose"},
        {"category":"Coral","brand":"L'Oréal","shade":"Rosy Outlook","hex":"D86878","desc":"Pink coral"},
        {"category":"Bold","brand":"MAC","shade":"Diva","hex":"4A1830","desc":"Dark burgundy"},
        {"category":"Bold","brand":"NYX","shade":"Strawberry Milk","hex":"C05068","desc":"Berry red"},
    ],
    ("Light","Neutral"):[
        {"category":"Nude","brand":"MAC","shade":"Spirit","hex":"C09070","desc":"Neutral nude"},
        {"category":"Nude","brand":"Maybelline","shade":"Warm Me Up","hex":"B88060","desc":"Natural warm"},
        {"category":"Coral","brand":"L'Oréal","shade":"Peach Fuzz","hex":"E07858","desc":"Peachy coral"},
        {"category":"Coral","brand":"Revlon","shade":"Apricot Nectar","hex":"D87858","desc":"Muted peach"},
        {"category":"Bold","brand":"MAC","shade":"Sin","hex":"68203A","desc":"Deep plum berry"},
        {"category":"Bold","brand":"NYX","shade":"Siren","hex":"8C1830","desc":"True red berry"},
    ],
    ("Intermediate","Warm"):[
        {"category":"Nude","brand":"MAC","shade":"Café Latte","hex":"A86840","desc":"Warm brown nude"},
        {"category":"Nude","brand":"NYX","shade":"Nude Suede Shoes","hex":"9A6038","desc":"Terracotta nude"},
        {"category":"Coral","brand":"L'Oréal","shade":"Spiced Coral","hex":"C85838","desc":"Spiced orange coral"},
        {"category":"Coral","brand":"Maybelline","shade":"Coral Crush","hex":"D06038","desc":"Rich coral"},
        {"category":"Bold","brand":"MAC","shade":"All Fired Up","hex":"A01828","desc":"Hot warm red"},
        {"category":"Bold","brand":"NYX","shade":"Crazed","hex":"5A1018","desc":"Deep warm berry"},
    ],
    ("Intermediate","Cool"):[
        {"category":"Nude","brand":"MAC","shade":"Blankety","hex":"B08070","desc":"Dusty rose nude"},
        {"category":"Nude","brand":"Revlon","shade":"Mauve It Over","hex":"987068","desc":"Cool mauve"},
        {"category":"Coral","brand":"L'Oréal","shade":"Fuchsia Flush","hex":"C04870","desc":"Cool pink coral"},
        {"category":"Coral","brand":"NYX","shade":"Temptress","hex":"983060","desc":"Berry coral"},
        {"category":"Bold","brand":"MAC","shade":"Violetta","hex":"602848","desc":"Purple berry"},
        {"category":"Bold","brand":"Maybelline","shade":"Plum Please","hex":"481838","desc":"Deep plum"},
    ],
    ("Intermediate","Neutral"):[
        {"category":"Nude","brand":"MAC","shade":"Mocha","hex":"986050","desc":"Mid-tone nude"},
        {"category":"Nude","brand":"NYX","shade":"Whipped Caviar","hex":"886050","desc":"Natural brown"},
        {"category":"Coral","brand":"Revlon","shade":"Terra Cotta","hex":"C06040","desc":"Earthy coral"},
        {"category":"Coral","brand":"L'Oréal","shade":"Sienna Spice","hex":"B85838","desc":"Spiced peach"},
        {"category":"Bold","brand":"MAC","shade":"Chili","hex":"882018","desc":"Deep warm red"},
        {"category":"Bold","brand":"NYX","shade":"Cannes","hex":"701828","desc":"Rich berry red"},
    ],
    ("Tan","Warm"):[
        {"category":"Nude","brand":"MAC","shade":"Bronx","hex":"885030","desc":"Deep warm nude"},
        {"category":"Nude","brand":"NYX","shade":"Raw","hex":"784020","desc":"Rich brown nude"},
        {"category":"Coral","brand":"L'Oréal","shade":"Burnished Coral","hex":"B04820","desc":"Deep orange coral"},
        {"category":"Coral","brand":"Maybelline","shade":"Copper Spice","hex":"A84018","desc":"Copper coral"},
        {"category":"Bold","brand":"MAC","shade":"Marrakesh","hex":"781808","desc":"Deep brick red"},
        {"category":"Bold","brand":"NYX","shade":"Dirty Talk","hex":"601008","desc":"Oxblood"},
    ],
    ("Tan","Cool"):[
        {"category":"Nude","brand":"MAC","shade":"Taupe","hex":"906858","desc":"Cool brown nude"},
        {"category":"Nude","brand":"Revlon","shade":"Dusty Mauve","hex":"886060","desc":"Muted mauve"},
        {"category":"Coral","brand":"NYX","shade":"Athens","hex":"A84858","desc":"Berry coral"},
        {"category":"Coral","brand":"L'Oréal","shade":"Rose Mauve","hex":"A05060","desc":"Dusky rose"},
        {"category":"Bold","brand":"MAC","shade":"Cyber","hex":"401030","desc":"Dark plum"},
        {"category":"Bold","brand":"Maybelline","shade":"Midnight Plum","hex":"380820","desc":"Blackened plum"},
    ],
    ("Tan","Neutral"):[
        {"category":"Nude","brand":"MAC","shade":"Persistence","hex":"885848","desc":"Warm neutral nude"},
        {"category":"Nude","brand":"NYX","shade":"Stockholm","hex":"805040","desc":"Mid brown nude"},
        {"category":"Coral","brand":"Revlon","shade":"Rich Girl Red","hex":"A03828","desc":"Deep coral red"},
        {"category":"Coral","brand":"L'Oréal","shade":"Sandstorm","hex":"B05030","desc":"Sandy coral"},
        {"category":"Bold","brand":"MAC","shade":"Rebel","hex":"6B2D4E","desc":"Deep berry"},
        {"category":"Bold","brand":"NYX","shade":"Dante","hex":"580818","desc":"Dark red wine"},
    ],
    ("Brown","Warm"):[
        {"category":"Nude","brand":"MAC","shade":"Paramount","hex":"704028","desc":"Rich warm nude"},
        {"category":"Nude","brand":"NYX","shade":"Ginger Snap","hex":"603018","desc":"Spice nude"},
        {"category":"Coral","brand":"L'Oréal","shade":"Toasted Almond","hex":"983020","desc":"Deep terracotta"},
        {"category":"Coral","brand":"Maybelline","shade":"Copper Rose","hex":"883018","desc":"Copper coral"},
        {"category":"Bold","brand":"MAC","shade":"D For Danger","hex":"601010","desc":"Deep true red"},
        {"category":"Bold","brand":"NYX","shade":"Cabo","hex":"500808","desc":"Dark oxblood"},
    ],
    ("Brown","Cool"):[
        {"category":"Nude","brand":"MAC","shade":"Soar","hex":"785050","desc":"Cool brown-nude"},
        {"category":"Nude","brand":"Revlon","shade":"Berry Couture","hex":"704858","desc":"Muted berry nude"},
        {"category":"Coral","brand":"NYX","shade":"Milan","hex":"903858","desc":"Cool berry coral"},
        {"category":"Coral","brand":"L'Oréal","shade":"Plum Fusion","hex":"884060","desc":"Deep berry"},
        {"category":"Bold","brand":"MAC","shade":"Deeply Forbidden","hex":"380828","desc":"Deep blackberry"},
        {"category":"Bold","brand":"Maybelline","shade":"Berry Stain","hex":"300818","desc":"Dark berry"},
    ],
    ("Brown","Neutral"):[
        {"category":"Nude","brand":"MAC","shade":"Whirl","hex":"784840","desc":"Dirty rose nude"},
        {"category":"Nude","brand":"NYX","shade":"Half The World Away","hex":"704030","desc":"Warm brown nude"},
        {"category":"Coral","brand":"L'Oréal","shade":"Spice It Up","hex":"903028","desc":"Deep spiced coral"},
        {"category":"Coral","brand":"Revlon","shade":"Rich Coral","hex":"882820","desc":"Muted deep coral"},
        {"category":"Bold","brand":"MAC","shade":"Film Noir","hex":"301018","desc":"Dark wine"},
        {"category":"Bold","brand":"NYX","shade":"Moonset","hex":"280810","desc":"Blackened red"},
    ],
    ("Dark","Warm"):[
        {"category":"Nude","brand":"MAC","shade":"Cosmo","hex":"583018","desc":"Deep warm nude"},
        {"category":"Nude","brand":"NYX","shade":"Tiramisu","hex":"502010","desc":"Rich chocolate nude"},
        {"category":"Coral","brand":"L'Oréal","shade":"Volcanic Coral","hex":"782010","desc":"Deep orange-red"},
        {"category":"Coral","brand":"Maybelline","shade":"Brick Beat","hex":"681808","desc":"Brick coral"},
        {"category":"Bold","brand":"MAC","shade":"Viva Glam I","hex":"701010","desc":"Deep red"},
        {"category":"Bold","brand":"NYX","shade":"Africa","hex":"480808","desc":"Dark oxblood red"},
    ],
    ("Dark","Cool"):[
        {"category":"Nude","brand":"MAC","shade":"Antique Velvet","hex":"603040","desc":"Cool deep nude"},
        {"category":"Nude","brand":"Revlon","shade":"Dark Plum Nude","hex":"503040","desc":"Plum nude"},
        {"category":"Coral","brand":"NYX","shade":"Canberra","hex":"782848","desc":"Deep berry coral"},
        {"category":"Coral","brand":"L'Oréal","shade":"Deep Rose","hex":"702040","desc":"Deep fuchsia"},
        {"category":"Bold","brand":"MAC","shade":"Smoked Purple","hex":"301028","desc":"Dark purple"},
        {"category":"Bold","brand":"Maybelline","shade":"Midnight Berry","hex":"280818","desc":"Blackened berry"},
    ],
    ("Dark","Neutral"):[
        {"category":"Nude","brand":"MAC","shade":"Stone","hex":"583830","desc":"Neutral deep nude"},
        {"category":"Nude","brand":"NYX","shade":"Moonwalk","hex":"503028","desc":"Deep neutral"},
        {"category":"Coral","brand":"L'Oréal","shade":"Copper Blaze","hex":"702010","desc":"Deep copper"},
        {"category":"Coral","brand":"Revlon","shade":"Spicy Cinnamon","hex":"681810","desc":"Spiced red"},
        {"category":"Bold","brand":"MAC","shade":"Lorde","hex":"401820","desc":"Deep wine"},
        {"category":"Bold","brand":"NYX","shade":"Rome","hex":"380810","desc":"Dark garnet"},
    ],
}

def get_lipsticks(skin_tone, undertone):
    return LIPSTICK_TABLE.get((skin_tone, undertone),
           LIPSTICK_TABLE.get((skin_tone, "Neutral"), []))

# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════
DATASET_PATH = "shades.csv"
df = knn = None

@app.on_event("startup")
def load_data():
    global df, knn
    if not os.path.exists(DATASET_PATH):
        print(f"WARNING: {DATASET_PATH} not found!"); return
    df = pd.read_csv(DATASET_PATH)
    df.dropna(subset=["L","A","B_lab"], inplace=True)
    knn = NearestNeighbors(n_neighbors=15, metric="euclidean")
    knn.fit(df[["L","A","B_lab"]].values)
    print(f"Loaded {len(df)} shades from {df['brand'].nunique()} brands")
    print(f"Columns: {list(df.columns)}")

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def rgb_to_lab(r,g,b):
    r,g,b=r/255.,g/255.,b/255.
    def lin(c): return ((c+0.055)/1.055)**2.4 if c>0.04045 else c/12.92
    r,g,b=lin(r),lin(g),lin(b)
    X=(r*0.4124564+g*0.3575761+b*0.1804375)/0.95047
    Y=(r*0.2126729+g*0.7151522+b*0.0721750)/1.
    Z=(r*0.0193339+g*0.1191920+b*0.9503041)/1.08883
    def f(t): return t**(1/3) if t>0.008856 else 7.787*t+16/116
    fx,fy,fz=f(X),f(Y),f(Z)
    return round(116*fy-16,3),round(500*(fx-fy),3),round(200*(fy-fz),3)

def get_undertone(a,b):
    if b>18: return "Warm"
    elif b<10 or a>12: return "Cool"
    return "Neutral"

def get_skin_tone(ita):
    if ita>55: return "Very Light"
    elif ita>41: return "Light"
    elif ita>28: return "Intermediate"
    elif ita>10: return "Tan"
    elif ita>-30: return "Brown"
    return "Dark"


# ═══════════════════════════════════════════════════════════════
# EXACT SHOP URL TABLE
# (brand, product_short, shade_number) → direct product URL
# ═══════════════════════════════════════════════════════════════
NYKAA_BASE = "https://www.nykaa.com/search/result/?q="

EXACT_URLS = {
    ("Maybelline","fmf","102"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8082",
    ("Maybelline","fmf","110"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8083",
    ("Maybelline","fmf","115"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8084",
    ("Maybelline","fmf","120"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8085",
    ("Maybelline","fmf","125"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8086",
    ("Maybelline","fmf","128"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8087",
    ("Maybelline","fmf","130"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8088",
    ("Maybelline","fmf","220"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8089",
    ("Maybelline","fmf","230"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8090",
    ("Maybelline","fmf","240"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8091",
    ("Maybelline","fmf","310"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8092",
    ("Maybelline","fmf","320"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8093",
    ("Maybelline","fmf","330"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8094",
    ("Maybelline","fmf","340"): "https://www.nykaa.com/maybelline-new-york-fit-me-matte-poreless-foundation/p/1076?skuId=8095",
    ("MAC","sff","NC5"):  "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NC5",
    ("MAC","sff","NC10"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NC10",
    ("MAC","sff","NC15"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NC15",
    ("MAC","sff","NC20"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NC20",
    ("MAC","sff","NC25"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NC25",
    ("MAC","sff","NC30"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NC30",
    ("MAC","sff","NC35"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NC35",
    ("MAC","sff","NC40"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NC40",
    ("MAC","sff","NC45"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NC45",
    ("MAC","sff","NC50"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NC50",
    ("MAC","sff","NW10"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NW10",
    ("MAC","sff","NW15"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NW15",
    ("MAC","sff","NW20"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NW20",
    ("MAC","sff","NW25"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NW25",
    ("MAC","sff","NW30"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NW30",
    ("MAC","sff","NW35"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NW35",
    ("MAC","sff","NW40"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NW40",
    ("MAC","sff","NW45"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NW45",
    ("MAC","sff","NW50"): "https://www.maccosmetics.com/product/13847/58380/products/makeup/face/foundation/studio-fix-fluid-spf-15-foundation#shade=NW50",
    ("NARS","nrl","OSLO"):       "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080695.html",
    ("NARS","nrl","DEAUVILLE"):  "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080718.html",
    ("NARS","nrl","MONT BLANC"): "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080725.html",
    ("NARS","nrl","SYRACUSE"):   "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080732.html",
    ("NARS","nrl","BARCELONA"):  "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080756.html",
    ("NARS","nrl","TAHOE"):      "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080763.html",
    ("NARS","nrl","MARRAKESH"):  "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080800.html",
    ("NARS","nrl","MACAO"):      "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080817.html",
    ("NARS","nrl","ISTANBUL"):   "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080824.html",
    ("NARS","nrl","BALI"):       "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080848.html",
    ("NARS","nrl","MARTINIQUE"): "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080862.html",
    ("NARS","nrl","CEYLAN"):     "https://www.narscosmetics.com/natural-radiant-longwear-foundation/0607845080909.html",
    ("Fenty Beauty","pf","100W"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=100W",
    ("Fenty Beauty","pf","110N"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=110N",
    ("Fenty Beauty","pf","120W"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=120W",
    ("Fenty Beauty","pf","130W"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=130W",
    ("Fenty Beauty","pf","140W"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=140W",
    ("Fenty Beauty","pf","150W"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=150W",
    ("Fenty Beauty","pf","160W"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=160W",
    ("Fenty Beauty","pf","170N"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=170N",
    ("Fenty Beauty","pf","180W"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=180W",
    ("Fenty Beauty","pf","200W"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=200W",
    ("Fenty Beauty","pf","220N"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=220N",
    ("Fenty Beauty","pf","240N"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=240N",
    ("Fenty Beauty","pf","260N"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=260N",
    ("Fenty Beauty","pf","290N"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=290N",
    ("Fenty Beauty","pf","498N"): "https://fentybeauty.com/products/pro-filtr-soft-matte-longwear-foundation?variant=498N",
    ("Revlon","cs","110"): "https://www.nykaa.com/revlon-colorstay-foundation/p/2788?skuId=2789",
    ("Revlon","cs","120"): "https://www.nykaa.com/revlon-colorstay-foundation/p/2788?skuId=2790",
    ("Revlon","cs","150"): "https://www.nykaa.com/revlon-colorstay-foundation/p/2788?skuId=2791",
    ("Revlon","cs","180"): "https://www.nykaa.com/revlon-colorstay-foundation/p/2788?skuId=2792",
    ("Revlon","cs","200"): "https://www.nykaa.com/revlon-colorstay-foundation/p/2788?skuId=2793",
    ("Revlon","cs","220"): "https://www.nykaa.com/revlon-colorstay-foundation/p/2788?skuId=2794",
    ("Revlon","cs","300"): "https://www.nykaa.com/revlon-colorstay-foundation/p/2788?skuId=2796",
    ("Revlon","cs","330"): "https://www.nykaa.com/revlon-colorstay-foundation/p/2788?skuId=2797",
    ("Revlon","cs","370"): "https://www.nykaa.com/revlon-colorstay-foundation/p/2788?skuId=2798",
    ("Revlon","cs","390"): "https://www.nykaa.com/revlon-colorstay-foundation/p/2788?skuId=2799",
    ("Bobbi Brown","sf","0"):  "https://www.nykaa.com/bobbi-brown-skin-foundation-spf-15/p/6993?skuId=6993",
    ("Bobbi Brown","sf","1"):  "https://www.nykaa.com/bobbi-brown-skin-foundation-spf-15/p/6993?skuId=6995",
    ("Bobbi Brown","sf","2"):  "https://www.nykaa.com/bobbi-brown-skin-foundation-spf-15/p/6993?skuId=6996",
    ("Bobbi Brown","sf","3"):  "https://www.nykaa.com/bobbi-brown-skin-foundation-spf-15/p/6993?skuId=6997",
    ("Bobbi Brown","sf","4"):  "https://www.nykaa.com/bobbi-brown-skin-foundation-spf-15/p/6993?skuId=6998",
    ("Bobbi Brown","sf","5"):  "https://www.nykaa.com/bobbi-brown-skin-foundation-spf-15/p/6993?skuId=6999",
    ("Bobbi Brown","sf","6"):  "https://www.nykaa.com/bobbi-brown-skin-foundation-spf-15/p/6993?skuId=7000",
    ("Bobbi Brown","sf","7"):  "https://www.nykaa.com/bobbi-brown-skin-foundation-spf-15/p/6993?skuId=7001",
}

LIP_EXACT_URLS = {
    ("MAC","Velvet Teddy"):   "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Velvet_Teddy",
    ("MAC","Ruby Woo"):       "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Ruby_Woo",
    ("MAC","Lady Danger"):    "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Lady_Danger",
    ("MAC","Russian Red"):    "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Russian_Red",
    ("MAC","Rebel"):          "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Rebel",
    ("MAC","Diva"):           "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Diva",
    ("MAC","Honey Love"):     "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Honey_Love",
    ("MAC","Twig"):           "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Twig",
    ("MAC","Sin"):            "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Sin",
    ("MAC","Chili"):          "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Chili",
    ("MAC","Marrakesh"):      "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Marrakesh",
    ("MAC","Cyber"):          "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Cyber",
    ("MAC","Whirl"):          "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Whirl",
    ("MAC","Film Noir"):      "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Film_Noir",
    ("MAC","Viva Glam I"):    "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Viva_Glam_I",
    ("MAC","All Fired Up"):   "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=All_Fired_Up",
    ("MAC","Violetta"):       "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Violetta",
    ("MAC","D For Danger"):   "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=D_For_Danger",
    ("MAC","Cosmo"):          "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Cosmo",
    ("MAC","Lorde"):          "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Lorde",
    ("MAC","Stone"):          "https://www.maccosmetics.com/product/13854/310/products/makeup/lips/lipstick/matte-lipstick#shade=Stone",
    ("NYX","Alabama"):        "https://www.nykaa.com/search/result/?q=NYX+Soft+Matte+Lip+Cream+Alabama",
    ("NYX","Bordeaux"):       "https://www.nykaa.com/search/result/?q=NYX+Soft+Matte+Lip+Cream+Bordeaux",
    ("NYX","Dark Era"):       "https://www.nykaa.com/search/result/?q=NYX+Matte+Lipstick+Dark+Era",
    ("NYX","Crazed"):         "https://www.nykaa.com/search/result/?q=NYX+Matte+Lipstick+Crazed",
    ("NYX","Siren"):          "https://www.nykaa.com/search/result/?q=NYX+Matte+Lipstick+Siren",
    ("NYX","Cannes"):         "https://www.nykaa.com/search/result/?q=NYX+Soft+Matte+Lip+Cream+Cannes",
    ("NYX","Dante"):          "https://www.nykaa.com/search/result/?q=NYX+Soft+Matte+Lip+Cream+Dante",
    ("NYX","Cabo"):           "https://www.nykaa.com/search/result/?q=NYX+Soft+Matte+Lip+Cream+Cabo",
    ("NYX","Milan"):          "https://www.nykaa.com/search/result/?q=NYX+Soft+Matte+Lip+Cream+Milan",
    ("NYX","Rome"):           "https://www.nykaa.com/search/result/?q=NYX+Soft+Matte+Lip+Cream+Rome",
    ("L'Oréal","Coral Bistro"):   "https://www.nykaa.com/search/result/?q=LOreal+Color+Riche+Coral+Bistro",
    ("L'Oréal","Sunset Coral"):   "https://www.nykaa.com/search/result/?q=LOreal+Color+Riche+Sunset+Coral",
    ("L'Oréal","Tickled Pink"):   "https://www.nykaa.com/search/result/?q=LOreal+Color+Riche+Tickled+Pink",
    ("L'Oréal","Peach Fuzz"):     "https://www.nykaa.com/search/result/?q=LOreal+Color+Riche+Peach+Fuzz",
    ("L'Oréal","Spiced Coral"):   "https://www.nykaa.com/search/result/?q=LOreal+Color+Riche+Spiced+Coral",
    ("L'Oréal","Toasted Almond"): "https://www.nykaa.com/search/result/?q=LOreal+Color+Riche+Toasted+Almond",
    ("Maybelline","Peach Pop"):    "https://www.nykaa.com/search/result/?q=Maybelline+Color+Sensational+Peach+Pop",
    ("Maybelline","Coral Spice"):  "https://www.nykaa.com/search/result/?q=Maybelline+Color+Sensational+Coral+Spice",
    ("Maybelline","Copper Rose"):  "https://www.nykaa.com/search/result/?q=Maybelline+Color+Sensational+Copper+Rose",
    ("Revlon","Pink Velvet"):      "https://www.nykaa.com/search/result/?q=Revlon+Super+Lustrous+Pink+Velvet",
    ("Revlon","Coral Berry"):      "https://www.nykaa.com/search/result/?q=Revlon+Super+Lustrous+Coral+Berry",
    ("Revlon","Rich Girl Red"):    "https://www.nykaa.com/search/result/?q=Revlon+Super+Lustrous+Rich+Girl+Red",
    ("Revlon","Spicy Cinnamon"):   "https://www.nykaa.com/search/result/?q=Revlon+Super+Lustrous+Spicy+Cinnamon",
}

def get_shop_url(brand, product_short, shade_number, shade_name):
    """Return (url, platform) for a foundation/concealer shade."""
    from urllib.parse import quote
    # 1. Try exact URL first
    key = (brand, product_short, str(shade_number))
    if key in EXACT_URLS:
        platform = "Nykaa" if "nykaa.com" in EXACT_URLS[key] else "Brand Site"
        return EXACT_URLS[key], platform
    # 2. Fallback: smart search
    q = quote(f"{brand} {shade_name or shade_number}")
    nykaa_brands = ["Maybelline","L'Oreal","Revlon","Bobbi Brown","bareMinerals",
                    "Lakme","Lakmé","Nykaa","Colorbar","Lotus Herbals",
                    "Blue Heaven","Bharat & Doris","Beauty Bakerie","Black Opal",
                    "Iman","House of Tara","Black Up","Nykaa"]
    if brand in nykaa_brands:
        return f"https://www.nykaa.com/search/result/?q={q}", "Nykaa"
    return f"https://www.nykaa.com/search/result/?q={q}", "Nykaa"

def get_lip_url(brand, shade):
    """Return (url, platform) for a lipstick shade."""
    from urllib.parse import quote
    key = (brand, shade)
    if key in LIP_EXACT_URLS:
        platform = "Brand Site" if "maccosmetics.com" in LIP_EXACT_URLS[key] else "Nykaa"
        return LIP_EXACT_URLS[key], platform
    q = quote(f"{brand} {shade} lipstick")
    return f"https://www.nykaa.com/search/result/?q={q}", "Nykaa"

def row_to_dict(row, delta):
    # Pull shade number and name — available after running enrich_shades.py
    shade_number = str(row.get("shade_number","—")).strip()
    shade_name   = str(row.get("shade_name","—")).strip()
    product      = str(row.get("product_short","")).strip()
    brand        = str(row.get("brand","")).strip()

    # If shade_number is missing/dash, build a readable fallback from product_s
    product_s = str(row.get("product_s","")).strip()
    if shade_number in ("—","nan",""):
        shade_number = product_s.upper() if product_s else "—"
    if shade_name in ("—","nan",""):
        shade_name = "Shade " + shade_number if shade_number != "—" else product

    prod_short = str(row.get("product_short","")).strip()
    shop_url, shop_platform = get_shop_url(brand, prod_short, shade_number, shade_name)
    return {
        "brand":         brand,
        "product":       product,
        "shade_number":  shade_number,
        "shade_name":    shade_name,
        "hex":           str(row.get("hex","")).lstrip("#"),
        "skin_tone":     str(row.get("skin_tone","")),
        "undertone":     str(row.get("undertone","")),
        "delta_E":       float(delta),
        "L":             float(row.get("L",0)),
        "shop_url":      shop_url,
        "shop_platform": shop_platform,
    }

def do_knn(L,A,B,top_n=8):
    d,i   = knn.kneighbors([[L,A,B]])
    found = [row_to_dict(df.iloc[idx], d[0][n]) for n,idx in enumerate(i[0][:top_n])]
    # Concealer: 1-2 shades lighter (L* +4)
    Lc    = min(L+4, 100)
    dc,ic = knn.kneighbors([[Lc,A,B]])
    conc  = [row_to_dict(df.iloc[idx], dc[0][n]) for n,idx in enumerate(ic[0][:top_n])]
    return found, conc

# ═══════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════
@app.post("/analyse")
async def analyse_selfie(file: UploadFile=File(...), top_n: int=8):
    if df is None: raise HTTPException(500,"Dataset not loaded")
    contents = await file.read()
    img_arr  = np.array(Image.open(io.BytesIO(contents)).convert("RGB"))
    img_bgr  = cv2.cvtColor(img_arr,cv2.COLOR_RGB2BGR)
    gray     = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
    fc = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    faces = fc.detectMultiScale(gray,1.1,5,minSize=(60,60))
    if len(faces)==0: raise HTTPException(400,"No face detected.")
    x,y,w,h = max(faces,key=lambda f:f[2]*f[3])
    regions = [
        img_bgr[y+int(h*.05):y+int(h*.22), x+int(w*.3):x+int(w*.7)],
        img_bgr[y+int(h*.45):y+int(h*.65), x+int(w*.1):x+int(w*.35)],
        img_bgr[y+int(h*.45):y+int(h*.65), x+int(w*.65):x+int(w*.9)],
    ]
    all_px=[]
    for r in regions:
        if r.size==0: continue
        px=r.reshape(-1,3).astype(float); br=px.mean(axis=1)
        px=px[(br>40)&(br<220)]
        if len(px): all_px.append(px)
    if not all_px: raise HTTPException(400,"Could not sample skin pixels.")
    combined=np.vstack(all_px); ab,ag,ar=combined.mean(axis=0)
    L,A,B=rgb_to_lab(ar,ag,ab)
    ita=float(np.degrees(np.arctan2(L-50,B)))
    st=get_skin_tone(ita); ut=get_undertone(A,B)
    found,conc=do_knn(L,A,B,top_n)
    return {"skin":{"L":L,"A":A,"B":B,"ita":round(ita,2),"skin_tone":st,"undertone":ut,"rgb":[int(ar),int(ag),int(ab)]},
            "foundation":found,"concealer":conc,"lipstick":get_lipsticks(st,ut)}

class LabInput(BaseModel):
    L:float; A:float; B:float; top_n:int=8

@app.post("/match")
def match_manual(body:LabInput):
    if df is None: raise HTTPException(500,"Dataset not loaded")
    ita=float(np.degrees(np.arctan2(body.L-50,body.B)))
    st=get_skin_tone(ita); ut=get_undertone(body.A,body.B)
    found,conc=do_knn(body.L,body.A,body.B,body.top_n)
    return {"skin":{"L":body.L,"A":body.A,"B":body.B,"ita":round(ita,2),"skin_tone":st,"undertone":ut,"rgb":None},
            "foundation":found,"concealer":conc,"lipstick":get_lipsticks(st,ut)}

class HexInput(BaseModel):
    hex:str

@app.post("/hex-to-lab")
def hex_to_lab_ep(body:HexInput):
    h=body.hex.lstrip("#")
    if len(h)!=6: raise HTTPException(400,"Invalid hex")
    r,g,b=int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    L,A,B=rgb_to_lab(r,g,b)
    return {"L":L,"A":A,"B":B,"rgb":[r,g,b]}