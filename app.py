# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io, os, math

st.set_page_config(page_title="Personal Color Analysis — RGB (Final)", layout="wide")

CSV_NAME = "dataset_rgb.csv"
if not os.path.exists(CSV_NAME):
    st.error(f"Dataset not found: {CSV_NAME}. Place dataset_rgb.csv in app folder.")
    st.stop()

df = pd.read_csv(CSV_NAME)

expected = {"ID","SkinTone_Y","SkinTone_Cb","SkinTone_Cr","R","G","B","HEX","Label","ColorName"}
if not expected.issubset(set(df.columns)):
    st.error("CSV missing required columns. Expected: ID,SkinTone_Y,SkinTone_Cb,SkinTone_Cr,R,G,B,HEX,Label,ColorName")
    st.stop()

# -------------------------
# Helpers: color conversions
# -------------------------
def hex_to_rgb(hexcode):
    h = str(hexcode).lstrip('#')
    return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

def rgb_to_hex(r,g,b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def rgb_to_ycbcr_bt601_fullrange(r,g,b):
    # use same formula used earlier (approx BT.601 full range)
    y  =  16 + (65.738*r + 129.057*g + 25.064*b)/256
    cb = 128 + (-37.945*r - 74.494*g + 112.439*b)/256
    cr = 128 + (112.439*r - 94.154*g - 18.285*b)/256
    return float(y), float(cb), float(cr)

# -------------------------
# Face detection + skin extraction
# -------------------------
def pil_to_cv(p):
    return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)

def cv_to_pil(c):
    return Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))

def detect_face_box(cv_img):
    gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
    cas=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    faces=cas.detectMultiScale(gray,1.1,4,minSize=(60,60))
    if len(faces)==0:
        return None
    x,y,w,h = sorted(faces,key=lambda r:r[2]*r[3],reverse=True)[0]
    pad=int(w*0.13)
    return (max(0,x-pad), max(0,y-pad), w+2*pad, h+2*pad)

def skin_mask_roi(roi):
    # roi expected in BGR (cv image)
    ycc=cv2.cvtColor(roi,cv2.COLOR_BGR2YCrCb)
    Y,Cr,Cb = ycc[:,:,0], ycc[:,:,1], ycc[:,:,2]
    # relaxed bounds to work in varied lighting, then morphological cleanup
    mask = ((Cb>=70)&(Cb<=140)&(Cr>=135)&(Cr<=200)).astype(np.uint8)*255
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask,5)
    return mask

def median_skin(roi,mask):
    ycc=cv2.cvtColor(roi,cv2.COLOR_BGR2YCrCb)
    if mask is None or mask.sum()==0:
        return (int(np.median(ycc[:,:,0])), int(np.median(ycc[:,:,2])), int(np.median(ycc[:,:,1])))
    ys = ycc[:,:,0][mask==255]
    crs = ycc[:,:,1][mask==255]
    cbs = ycc[:,:,2][mask==255]
    return (int(np.median(ys)), int(np.median(cbs)), int(np.median(crs)))

# -------------------------
# Skin classification to avoid fair/wheat mistakes
# -------------------------
def classify_skin_tone(Y, Cb, Cr):
    diff = Cr - Cb
    if Y > 165 and diff < 18:
        return "Fair"
    if (150 < Y <= 170) or (18 <= diff <= 30):
        return "Medium"
    return "Dark"

# -------------------------
# nearest skin ID (based on SkinTone YCbCr in dataset)
# -------------------------
def nearest_skin(Y,Cb,Cr,df):
    uniq = df[["ID","SkinTone_Y","SkinTone_Cb","SkinTone_Cr"]].drop_duplicates().reset_index(drop=True)
    d = np.sqrt((uniq.SkinTone_Y - Y)**2 + (uniq.SkinTone_Cb - Cb)**2 + (uniq.SkinTone_Cr - Cr)**2)
    i = int(np.argmin(d.values))
    ID = int(uniq.loc[i,"ID"])
    return ID

# -------------------------
# Face metrics for comparison
# -------------------------
def face_metrics(face_pil):
    arr = np.array(face_pil)
    if arr.size == 0:
        return 0.0,0.0,0.0,0.0,0.0
    # convert rgb arrays
    y = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)[:,:,0].mean()
    s = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[:,:,1].mean()
    c = arr.std()
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    A = lab[:,:,1].mean()
    B = lab[:,:,2].mean()
    return float(y), float(s), float(c), float(A), float(B)

# -------------------------
# Build drape with RGB
# -------------------------
def build_drape(img_pil, box, rgb):
    # img_pil: PIL RGB
    W,H = img_pil.size
    x,y,w,h = box
    dr_y = y + h - int(H*0.05)
    dr_y = max(0, dr_y)
    dr_h = H - dr_y
    base = img_pil.copy().convert("RGBA")
    drape = Image.new("RGBA", (W, dr_h), (int(rgb[0]), int(rgb[1]), int(rgb[2]), 255))
    base.paste(drape, (0, dr_y), drape)
    return base.convert("RGB")

# -------------------------
# Scoring system (same idea as before, tuned for RGB)
# -------------------------
def score_color(orig, aft, is_best, is_worst):
    if is_best:
        return ("YES — This is your BEST color 💖", "Always flattering", ["Glow","Clear skin"])
    if is_worst:
        return ("NO — This is your WORST color ❌", "Not suitable", ["Dull","Emphasises shadows"])
    oY,oS,oC,oA,oB = orig
    aY,aS,aC,aA,aB = aft
    dY = aY - oY
    dS = aS - oS
    dC = aC - oC
    dA = aA - oA
    dB = aB - oB
    score = 0
    if dY > 1: score += 2
    if dS > 2: score += 2
    if dC > 0.8: score += 1
    if abs(dA) < 5 and abs(dB) < 5: score += 1
    if score >= 5:
        return ("YES — suits you","Bright & Vibrant",["Glow","Sharper"])
    elif score >= 3:
        return ("OK — Moderately Good","Soft Effect",["Subtle","Natural"])
    else:
        return ("NO — Not Suitable","Dulls Features",["Flat","Low contrast"])

# -------------------------
# Season mapping from HEX (HSV based)
# -------------------------
def season_of_hex(hx):
    r,g,b = hex_to_rgb(hx)
    hsv = cv2.cvtColor(np.uint8([[[r,g,b]]]), cv2.COLOR_RGB2HSV)[0][0]
    h = float(hsv[0]) * 2        # 0-360
    s = float(hsv[1])            # 0-255
    v = float(hsv[2])            # 0-255
    # Spring: warm + bright + clear
    if (0 <= h <= 70 or 320 <= h <= 360) and s >= 110 and v >= 180:
        return "Spring"
    # Summer: cool + soft + light
    if 150 < h <= 250 and s <= 120 and v >= 170:
        return "Summer"
    # Autumn: warm + deep + muted
    if (10 <= h <= 70) and s >= 90 and v <= 170:
        return "Autumn"
    # Winter: cool + bright / cool-deep
    if (250 < h <= 330) and s >= 130:
        return "Winter"
    if (h <= 20 or h >= 330) and v <= 160 and s >= 120:
        return "Winter"
    if v >= 200 and s >= 100: return "Spring"
    if v >= 180 and s <= 110: return "Summer"
    if v <= 170 and s >= 100: return "Autumn"
    return "Winter"

# Build season_map (unique HEX per season from df)
SEASONS = ["Spring","Summer","Autumn","Winter"]
season_map = {s: [] for s in SEASONS}
unique_hex = {s:set() for s in SEASONS}
# collect up to 12 per season
for idx,row in df.iterrows():
    hx = row.HEX
    try:
        season = season_of_hex(hx)
    except:
        season = "Spring"
    if hx in unique_hex[season]: continue
    if len(season_map[season]) < 12:
        season_map[season].append(idx)
        unique_hex[season].add(hx)
# fill if any season short
for s in SEASONS:
    if len(season_map[s]) < 12:
        extras = df[df.HEX.apply(lambda x: season_of_hex(x)==s)].index.tolist()
        for idx in extras:
            if len(season_map[s]) >= 12: break
            if idx not in season_map[s]:
                season_map[s].append(idx)

# -------------------------
# UI state
# -------------------------
st.title("Personal Color Analysis — PCA RGB (Final)")

if "selected" not in st.session_state:
    st.session_state.selected = []

if "last_img" not in st.session_state:
    st.session_state.last_img = None

# -------------------------
# Image input
# -------------------------
st.header("1) Upload or Take Photo")
cam = st.camera_input("Take Photo")
up = st.file_uploader("Or Upload Image", type=["jpg","jpeg","png"])

img = None
raw = None

if cam:
    raw = cam.getvalue()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
elif up:
    raw = up.getvalue()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

if raw and st.session_state.last_img != raw:
    st.session_state.selected = []
    st.session_state.last_img = raw

if img is None:
    st.stop()

small = img.resize((int(img.width*0.40), int(img.height*0.40)))
st.image(small)

cv = pil_to_cv(img)
box = detect_face_box(cv)
if box is None:
    st.warning("Face not detected. Try better lighting & straight face.")
    st.stop()

x,y,w,h = box
roi = cv[y:y+h, x:x+w]

mask = skin_mask_roi(roi)
Y, Cb, Cr = median_skin(roi, mask)

# slight adaptive brightness push for fairness separation
Y = float(min(255, Y * 1.05))

st.success(f"Skin detected: Y={Y:.1f} Cb={Cb:.1f} Cr={Cr:.1f}  — Category: {classify_skin_tone(Y,Cb,Cr)}")

ID = nearest_skin(Y,Cb,Cr,df)
matched_rows = df[df.ID==ID]

best = matched_rows[matched_rows.Label.str.lower()=="best"].head(5)
worst = matched_rows[matched_rows.Label.str.lower()=="worst"].head(5)

# -------------------------
# Best & Worst display
# -------------------------
st.header("2) Suggested Colors (Best & Worst)")

st.subheader("Best 5 Colors")
best_cols = st.columns(5)
for i in range(5):
    if i < len(best):
        r = best.iloc[i]
        hx = r.HEX
        rgb = (int(r.R), int(r.G), int(r.B))
        with best_cols[i]:
            st.markdown(f"<div style='height:60px;border-radius:8px;background:{hx}'></div>", unsafe_allow_html=True)
            if st.button(f"Select Best {i+1}", key=f"b{i}"):
                st.session_state.selected.append({"idx": int(r.name), "type":"best"})

st.subheader("Worst 5 Colors")
worst_cols = st.columns(5)
for i in range(5):
    if i < len(worst):
        r = worst.iloc[i]
        hx = r.HEX
        with worst_cols[i]:
            st.markdown(f"<div style='height:60px;border-radius:8px;background:{hx}'></div>", unsafe_allow_html=True)
            if st.button(f"Select Worst {i+1}", key=f"w{i}"):
                st.session_state.selected.append({"idx": int(r.name), "type":"worst"})

# -------------------------
# Seasonal palettes
# -------------------------
st.header("3) Seasonal Palettes (pick any color to test)")
for season in SEASONS:
    with st.expander(season):
        cols = st.columns(6)
        season_rows = season_map[season]
        for position, row_idx in enumerate(season_rows):
            row = df.loc[row_idx]
            hx = row.HEX
            col = cols[position % 6]
            with col:
                st.markdown(f"<div style='height:45px;border-radius:6px;background:{hx}'></div>", unsafe_allow_html=True)
                if st.button(f"Select {season} {position+1}", key=f"{season}_{position}"):
                    st.session_state.selected.append({"idx": int(row_idx), "type":"season"})

# -------------------------
# Preview section — side by side
# -------------------------
st.header("4) Color Preview — Side by Side Comparison")
face_before = img.crop((x, y, x+w, y+h))
orig_metrics = face_metrics(face_before)

if len(st.session_state.selected) == 0:
    st.info("Select any color above to preview how it affects your face.")
else:
    for sel in st.session_state.selected:
        if sel["idx"] not in df.index:
            continue
        row = df.loc[sel["idx"]]
        rgb = (int(row.R), int(row.G), int(row.B))
        hx = row.HEX
        after_img = build_drape(img, (x,y,w,h), rgb)
        face_after = after_img.crop((x, y, x+w, y+h))
        aft_metrics = face_metrics(face_after)
        is_best = sel["type"] == "best"
        is_worst = sel["type"] == "worst"
        verdict, subtitle, effects = score_color(orig_metrics, aft_metrics, is_best, is_worst)

        st.markdown(f"## {sel['type'].upper()} — {hx} — {row.ColorName}")
        c1, c2 = st.columns([1,1])

        with c1:
            st.write("### Original")
            st.image(face_before.resize((int(face_before.width*1.5), int(face_before.height*1.5))))
            st.write(f"Brightness: {orig_metrics[0]:.1f}")
            st.write(f"Saturation: {orig_metrics[1]:.1f}")
            st.write(f"Contrast: {orig_metrics[2]:.1f}")

        with c2:
            st.write("### With Selected Color")
            st.image(face_after.resize((int(face_after.width*1.5), int(face_after.height*1.5))))
            st.write(f"Brightness Δ {aft_metrics[0] - orig_metrics[0]:+.1f}")
            st.write(f"Saturation Δ {aft_metrics[1] - orig_metrics[1]:+.1f}")
            st.write(f"Contrast Δ {aft_metrics[2] - orig_metrics[2]:+.1f}")
            st.markdown(f"### Verdict: **{verdict}**")
            st.markdown(f"**{subtitle}**")
            st.write("Effects: " + ", ".join(effects))

# -------------------------
# Optional: clear selection
# -------------------------
if st.button("Clear selections"):
    st.session_state.selected = []
    st.success("Cleared")
