import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io, os, math

st.set_page_config(page_title="Personal Color Analysis", layout="wide")

CSV_NAME = "skin_tone_color_analysis_dataset.csv"
if not os.path.exists(CSV_NAME):
    st.error(f"CSV not found: {CSV_NAME}")
    st.stop()

df = pd.read_csv(CSV_NAME)
expected = {"ID","SkinTone_Y","SkinTone_Cb","SkinTone_Cr",
            "Color_Y","Color_Cb","Color_Cr","Label"}
if not expected.issubset(df.columns):
    st.error("CSV missing required columns.")
    st.stop()

# --------------------------------------------------------------
# UTILITIES
# --------------------------------------------------------------

def rgb_from_ycbcr(Y, Cb, Cr):
    y,cb,cr=float(Y),float(Cb),float(Cr)
    r=1.164*(y-16)+1.596*(cr-128)
    g=1.164*(y-16)-0.392*(cb-128)-0.813*(cr-128)
    b=1.164*(y-16)+2.017*(cb-128)
    arr=np.clip([r,g,b],0,255).astype(np.uint8)
    return int(arr[0]),int(arr[1]),int(arr[2])

def hex_from_ycbcr(Y,Cb,Cr):
    r,g,b=rgb_from_ycbcr(Y,Cb,Cr)
    return f"#{r:02x}{g:02x}{b:02x}"

def pil_to_cv(p): 
    return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)

def cv_to_pil(c): 
    return Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))

# --------------------------------------------------------------
# FACE DETECTION + SKIN EXTRACTION
# --------------------------------------------------------------

def detect_face_box(cv_img):
    gray=cv2.cvtColor(cv_img,cv2.COLOR_BGR2GRAY)
    cas=cv2.CascadeClassifier(cv2.data.haarcascades+
                              "haarcascade_frontalface_default.xml")
    faces=cas.detectMultiScale(gray,1.1,4,minSize=(60,60))
    if len(faces)==0: 
        return None
    
    x,y,w,h = sorted(faces,key=lambda r:r[2]*r[3],reverse=True)[0]
    pad=int(w*0.13)
    return (max(0,x-pad), max(0,y-pad), w+2*pad, h+2*pad)

def skin_mask_roi(roi):
    ycc=cv2.cvtColor(roi,cv2.COLOR_BGR2YCrCb)
    Y=ycc[:,:,0]; Cr=ycc[:,:,1]; Cb=ycc[:,:,2]
    mask=np.logical_and.reduce(
        (Y>40,Y<240,Cb>75,Cb<180,Cr>130,Cr<200)
    ).astype(np.uint8)*255
    return cv2.medianBlur(mask,5)

def median_skin(roi,mask):
    ycc=cv2.cvtColor(roi,cv2.COLOR_BGR2YCrCb)
    if mask.sum()==0:
        return (int(np.median(ycc[:,:,0])),
                int(np.median(ycc[:,:,2])),
                int(np.median(ycc[:,:,1])))

    return (
        int(np.median(ycc[:,:,0][mask==255])),
        int(np.median(ycc[:,:,2][mask==255])),
        int(np.median(ycc[:,:,1][mask==255]))
    )

def nearest_skin(Y,Cb,Cr,df):
    uniq=df[["ID","SkinTone_Y","SkinTone_Cb","SkinTone_Cr"]].drop_duplicates()
    d=np.sqrt((uniq.SkinTone_Y-Y)**2 + 
              (uniq.SkinTone_Cb-Cb)**2 +
              (uniq.SkinTone_Cr-Cr)**2)
    i=int(np.argmin(d))
    ID=int(uniq.iloc[i].ID)
    return ID, df[df.ID==ID]

# --------------------------------------------------------------
# FACE METRICS
# --------------------------------------------------------------

def face_metrics(face):
    arr=np.array(face)
    if arr.size==0: return 0,0,0,0,0

    y=cv2.cvtColor(arr,cv2.COLOR_RGB2YCrCb)[:,:,0].mean()
    s=cv2.cvtColor(arr,cv2.COLOR_RGB2HSV)[:,:,1].mean()
    c=arr.std()
    lab=cv2.cvtColor(arr,cv2.COLOR_RGB2LAB)
    A=lab[:,:,1].mean()
    B=lab[:,:,2].mean()
    return float(y),float(s),float(c),float(A),float(B)

# --------------------------------------------------------------
# DRAPE
# --------------------------------------------------------------

def build_drape(img,box,rgb):
    W,H=img.size
    x,y,w,h = box
    dr_y = y+h - int(H*0.05)
    dr_y = max(0,dr_y)
    dr_h = H - dr_y

    base = img.copy().convert("RGBA")
    drape = Image.new("RGBA",(W,dr_h),(rgb[0],rgb[1],rgb[2],255))
    base.paste(drape,(0,dr_y),drape)
    return base.convert("RGB")
# --------------------------------------------------------------
# SCORING SYSTEM (BEST / WORST guaranteed + PCA logic)
# --------------------------------------------------------------

def score_color(orig, aft, is_best, is_worst):

    # FORCE BEST RESULT
    if is_best:
        return ("YES — This is your BEST color 💖",
                "Always flattering",
                ["Glow","Clear skin"])

    # FORCE WORST RESULT
    if is_worst:
        return ("NO — This is your WORST color ❌",
                "Not suitable",
                ["Dull","Emphasises shadows"])

    # PCA scoring for all other colors
    oB,oS,oC,oA,oBb = orig
    B,S,C,A,Bb = aft

    dB=B-oB
    dS=S-oS
    dC=C-oC
    dA=A-oA
    dBl=Bb-oBb

    score = 0
    if dB>1: score+=2
    if dS>1: score+=2
    if dC>0.8: score+=1
    if abs(dA)<5 and abs(dBl)<5: score+=1

    if score>=5:
        return ("YES — suits you",
                "Bright & Vibrant",
                ["Glow","Sharper"])
    elif score>=3:
        return ("OK — Moderately Good",
                "Soft Effect",
                ["Subtle","Natural"])
    else:
        return ("NO — Not Suitable",
                "Dulls Features",
                ["Flat","Low contrast"])


# --------------------------------------------------------------
# SEASON CATEGORY (Spring, Summer, Autumn, Winter)
# --------------------------------------------------------------

def season_of_row(row):
    r,g,b = rgb_from_ycbcr(row.Color_Y, row.Color_Cb, row.Color_Cr)
    hsv = cv2.cvtColor(np.uint8([[[r,g,b]]]), cv2.COLOR_RGB2HSV)[0][0]

    h = float(hsv[0]) * 2            # convert OpenCV 0–179 to 0–360
    s = float(hsv[1])                # saturation 0–255
    v = float(hsv[2])                # value 0–255

    # --- SPRING: warm + bright + clear ---
    if (0 <= h <= 70 or 320 <= h <= 360) and s >= 120 and v >= 180:
        return "Spring"

    # --- SUMMER: cool + soft + light ---
    if 150 < h <= 250 and s <= 120 and v >= 170:
        return "Summer"

    # --- AUTUMN: warm + deep + muted ---
    if (10 <= h <= 70) and s >= 100 and v <= 170:
        return "Autumn"

    # --- WINTER: cool + bright / cool-deep ---
    if (250 < h <= 330) and s >= 140:
        return "Winter"
    if (h <= 20 or h >= 330) and v <= 160 and s >= 120:
        return "Winter"

    # --- fallback by saturation/value (very accurate) ---
    if v >= 200 and s >= 100: return "Spring"
    if v >= 180 and s <= 110: return "Summer"
    if v <= 170 and s >= 100: return "Autumn"
    return "Winter"



# --------------------------------------------------------------
# BUILD SEASON MAP (20 colors each)
# --------------------------------------------------------------
# --------------------------------------------------------------
# BUILD SEASON MAP — 18 UNIQUE COLORS EACH
# --------------------------------------------------------------

SEASONS = ["Spring","Summer","Autumn","Winter"]
season_map = {s: [] for s in SEASONS}

# Track unique HEX so we don't repeat similar colors
unique_hex = {s: set() for s in SEASONS}

# First pass — take only UNIQUE colors per season (up to 18)
for idx, row in df.iterrows():
    season = season_of_row(row)
    hx = hex_from_ycbcr(row.Color_Y, row.Color_Cb, row.Color_Cr)

    # Skip if this HEX already added for this season
    if hx in unique_hex[season]:
        continue

    # Limit to 18 distinct colors
    if len(season_map[season]) < 10:
        season_map[season].append(idx)
        unique_hex[season].add(hx)

# Second pass — if any season has <18 colors, fill extra rows
for season in SEASONS:
    if len(season_map[season]) < 10:
        extra_rows = df[df.apply(lambda r: season_of_row(r) == season, axis=1)].index
        for idx in extra_rows:
            if len(season_map[season]) >= 10:
                break
            if idx not in season_map[season]:
                season_map[season].append(idx)



# --------------------------------------------------------------
# UI SETUP
# --------------------------------------------------------------

st.title("Personal Color Analysis — PCA Multi-Color System")

if "selected" not in st.session_state:
    st.session_state.selected = []

if "last_img" not in st.session_state:
    st.session_state.last_img = None


# --------------------------------------------------------------
# IMAGE INPUT
# --------------------------------------------------------------

st.header("1) Upload or Take Photo")

cam = st.camera_input("Take Photo")
up = st.file_uploader("Or Upload Image")

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

# SMALL DISPLAY
small = img.resize((int(img.width * 0.40), int(img.height * 0.40)))
st.image(small)


# --------------------------------------------------------------
# FACE DETECTION
# --------------------------------------------------------------

cv = pil_to_cv(img)
box = detect_face_box(cv)

if box is None:
    st.warning("Face not detected. Try better lighting & straight face.")
    st.stop()

x,y,w,h = box
roi = cv[y:y+h, x:x+w]

mask = skin_mask_roi(roi)
Y,Cb,Cr = median_skin(roi,mask)

st.success(f"Skin tone detected: Y={Y} Cb={Cb} Cr={Cr}")


# --------------------------------------------------------------
# MATCH BEST/WORST FROM DATASET
# --------------------------------------------------------------

ID, matched_rows = nearest_skin(Y,Cb,Cr,df)

best = matched_rows[matched_rows.Label.str.lower()=="best"].head(5)
worst = matched_rows[matched_rows.Label.str.lower()=="worst"].head(5)
# --------------------------------------------------------------
# 2) BEST & WORST COLORS SECTION
# --------------------------------------------------------------

st.header("2) Suggested Colors (Best & Worst)")

# ---------------- BEST COLORS ----------------
st.subheader("Best 5 Colors")
best_cols = st.columns(5)

for i in range(5):
    if i < len(best):
        r = best.iloc[i]
        hx = hex_from_ycbcr(r.Color_Y, r.Color_Cb, r.Color_Cr)

        with best_cols[i]:
            st.markdown(
                f"<div style='height:60px;border-radius:8px;background:{hx}'></div>",
                unsafe_allow_html=True
            )
            if st.button(f"Select Best {i+1}"):
                st.session_state.selected.append({
                    "idx": int(r.name),
                    "type": "best"
                })


# ---------------- WORST COLORS ----------------
st.subheader("Worst 5 Colors")
worst_cols = st.columns(5)

for i in range(5):
    if i < len(worst):
        r = worst.iloc[i]
        hx = hex_from_ycbcr(r.Color_Y, r.Color_Cb, r.Color_Cr)

        with worst_cols[i]:
            st.markdown(
                f"<div style='height:60px;border-radius:8px;background:{hx}'></div>",
                unsafe_allow_html=True
            )
            if st.button(f"Select Worst {i+1}"):
                st.session_state.selected.append({
                    "idx": int(r.name),
                    "type": "worst"
                })


# --------------------------------------------------------------
# 3) SEASONAL PALETTE (Spring / Summer / Autumn / Winter)
# --------------------------------------------------------------

st.header("3) Seasonal Palettes (Choose any colors to test)")

for season in SEASONS:
    with st.expander(season):

        cols = st.columns(6)
        season_rows = season_map[season]

        for position, row_idx in enumerate(season_rows):
            row = df.loc[row_idx]
            hx = hex_from_ycbcr(row.Color_Y, row.Color_Cb, row.Color_Cr)

            col = cols[position % 6]
            with col:
                st.markdown(
                    f"<div style='height:45px;border-radius:6px;background:{hx}'></div>",
                    unsafe_allow_html=True
                )
                if st.button(f"Select {season} {position+1}"):
                    st.session_state.selected.append({
                        "idx": row_idx,
                        "type": "season"
                    })
# --------------------------------------------------------------
# 4) PREVIEW SECTION — SIDE BY SIDE
# --------------------------------------------------------------

st.header("4) Color Preview — Side by Side Comparison")

# ORIGINAL FACE METRICS
face_before = img.crop((x, y, x+w, y+h))
orig_metrics = face_metrics(face_before)

if len(st.session_state.selected) == 0:
    st.info("Select any color above to preview how it affects your face.")
else:
    # For each selected color → show preview
    for sel in st.session_state.selected:

        if sel["idx"] not in df.index:
            continue

        row = df.loc[sel["idx"]]
        rgb = rgb_from_ycbcr(row.Color_Y, row.Color_Cb, row.Color_Cr)
        hx = hex_from_ycbcr(row.Color_Y, row.Color_Cb, row.Color_Cr)

        # Build drape
        after_img = build_drape(img, (x,y,w,h), rgb)

        # Crop face after drape
        face_after = after_img.crop((x, y, x+w, y+h))
        aft_metrics = face_metrics(face_after)

        # Score
        is_best = sel["type"] == "best"
        is_worst = sel["type"] == "worst"
        verdict, subtitle, effects = score_color(
            orig_metrics,
            aft_metrics,
            is_best,
            is_worst
        )

        # Title for each selected color
        st.markdown(f"## {sel['type'].upper()} — {hx}")

        # Two columns: Original vs Drape
        c1, c2 = st.columns([1,1])

        # ---- ORIGINAL IMAGE + METRICS ----
        with c1:
            st.write("### Original")
            st.image(
                img.resize((int(img.width*0.40), int(img.height*0.40))),
                use_container_width=False
            )
            st.write(f"Brightness: {orig_metrics[0]:.1f}")
            st.write(f"Saturation: {orig_metrics[1]:.1f}")
            st.write(f"Contrast: {orig_metrics[2]:.1f}")

        # ---- DRAPE IMAGE + METRICS ----
        with c2:
            st.write("### With Selected Color")
            st.image(
                after_img.resize((int(img.width*0.40), int(img.height*0.40))),
                use_container_width=False
            )
            st.write(f"Brightness Δ {aft_metrics[0] - orig_metrics[0]:+.1f}")
            st.write(f"Saturation Δ {aft_metrics[1] - orig_metrics[1]:+.1f}")
            st.write(f"Contrast Δ {aft_metrics[2] - orig_metrics[2]:+.1f}")

            st.markdown(f"### Verdict: **{verdict}**")
            st.markdown(f"**{subtitle}**")
            st.write("Effects: " + ", ".join(effects))

      







