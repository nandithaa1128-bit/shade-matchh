#  Shade Match

An AI-powered beauty web app that analyses your skin tone from a selfie and recommends the perfect **foundation**, **concealer**, and **lipstick** shades tailored to your skin tone and undertone.

---

##  Features

- **Selfie upload** — auto face detection samples skin tone from forehead and cheeks
- **Manual input** — Lab sliders or color picker for precise input
- **Exact shade numbers** — shows brand shade codes like `128 Warm Nude` or `NC25`
- **Direct shop links** — one click to buy on Nykaa or brand official website
- **Foundation + Concealer + Lipstick** — full beauty match in one place
- **625 shades across 36 brands** including Maybelline, MAC, NARS, Fenty Beauty, Revlon, Bobbi Brown, Dior and more

---

##  Project Structure

```
shade-match/
├── main.py              # FastAPI backend (skin analysis + KNN matching)
├── index.html           # Frontend (HTML/CSS/JS)
├── enrich_shades.py     # One-time script to add shade numbers to dataset
├── shades.csv           # Foundation shade dataset (625 shades, 36 brands)
├── requirements.txt     # Python dependencies
└── .gitignore
```

---

##  Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/shade-match.git
cd shade-match
```

### 2. Install dependencies
```bash
pip install fastapi uvicorn opencv-python scikit-learn pandas numpy pillow python-multipart
```

### 3. Enrich the dataset (run once)
```bash
python enrich_shades.py
```

### 4. Start the server
```bash
uvicorn main:app --reload
```

### 5. Open in browser
```
http://localhost:8000
```

---

## How It Works

1. **Face detection** — OpenCV Haar cascade detects your face
2. **Skin sampling** — Pixels from forehead and both cheeks are averaged
3. **Lab conversion** — RGB values are converted to CIE Lab color space
4. **ITA angle** — Individual Typology Angle classifies your skin tone category
5. **KNN matching** — K-Nearest Neighbours in Lab space finds closest foundation shades
6. **Concealer** — Derived by shifting L* +4 points lighter (industry standard)
7. **Lipstick** — Rule-based lookup by skin tone × undertone combination

---

##  Shopping Links

| Brand | Platform |
|---|---|
| Maybelline, Revlon, L'Oréal, NYX, Colorbar, Lakmé | 🛍 Nykaa (direct product page) |
| MAC, NARS, Fenty Beauty, Dior, Bobbi Brown | 🛒 Official brand website |

---

## 🛠️ Tech Stack

- **Backend** — Python, FastAPI, OpenCV, scikit-learn, pandas, numpy
- **Frontend** — Vanilla HTML/CSS/JavaScript
- **ML** — KNN (K-Nearest Neighbours) in CIE Lab color space
- **Dataset** — Kaggle Makeup Shades dataset (enriched with Lab + ITA + shade numbers)

---

## Dataset

The `shades.csv` dataset is sourced from Kaggle and enriched with:
- CIE Lab color values (L*, a*, b*)
- ITA angle and skin tone category
- Undertone classification (Warm / Cool / Neutral)
- Shade numbers and names per brand


---

## License

MIT License
---
## Author 
Nanditha A
