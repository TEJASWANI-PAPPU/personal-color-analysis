# app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io, os, math

st.set_page_config(page_title="PERSONAL COLOR ANALYSIS", layout="wide")

CSV_NAME = "colors_palette.csv"
if not os.path.exists(CSV_NAME):
    st.error(f"CSV not found: {CSV_NAME} — please place colors_palette.csv next to app.py")
    st.stop()

df = pd.read_csv(CSV_NAME)
expected = {"ID", "ColorName", "HEX", "Category"} if "Category" in pd.read_csv(CSV_NAME, nrows=0).columns else {"ID", "ColorName", "HEX"}
if not expected.issubset(set(df.columns)):
    st.error(f"CSV missing required columns. Expected at least: ID,ColorName,HEX. If you included Category column, it will be used.")
    st.stop()

# If CSV doesn't have Category, attempt to infer from known names (fallback: put all in 'Palette')
if "Category" not in df.columns:
    df["Category"] = "Palette"

# Helpers
def hex_to_rgb(hexcode):
    h = str(hexcode).strip().lstrip("#")
    if len(h) != 6:
        return (255,255,255)
    try:
        return (int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))
    except:
        return (255,255,255)

def pil_to_cv(p):
    return cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR)

def cv_to_pil(c):
    return Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))

def detect_face_box(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cas.detectMultiScale(gray, 1.1, 4, minSize=(60,60))
    if len(faces) == 0:
        return None
    x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
    pad = int(w * 0.13)
    return (max(0, x-pad), max(0, y-pad), w+2*pad, h+2*pad)

def skin_mask_roi(roi):
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = ycc[:,:,0], ycc[:,:,1], ycc[:,:,2]
    mask = ((Cb>=70)&(Cb<=140)&(Cr>=135)&(Cr<=200)).astype(np.uint8)*255
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask,5)
    return mask

def median_skin(roi, mask):
    ycc = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    if mask is None or mask.sum() == 0:
        return (int(np.median(ycc[:,:,0])), int(np.median(ycc[:,:,2])), int(np.median(ycc[:,:,1])))
    ys = ycc[:,:,0][mask==255]
    crs = ycc[:,:,1][mask==255]
    cbs = ycc[:,:,2][mask==255]
    return (int(np.median(ys)), int(np.median(cbs)), int(np.median(crs)))

def face_metrics(face_pil):
    arr = np.array(face_pil)
    if arr.size == 0:
        return 0.0,0.0,0.0,0.0,0.0
    y = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)[:,:,0].mean()
    s = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)[:,:,1].mean()
    c = arr.std()
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    A = lab[:,:,1].mean()
    B = lab[:,:,2].mean()
    return float(y), float(s), float(c), float(A), float(B)

def build_drape(img_pil, box, rgb):
    W,H = img_pil.size
    x,y,w,h = box
    dr_y = y + h - int(H*0.05)
    dr_y = max(0, dr_y)
    dr_h = H - dr_y
    base = img_pil.copy().convert("RGBA")
    drape = Image.new("RGBA", (W, dr_h), (int(rgb[0]), int(rgb[1]), int(rgb[2]), 255))
    base.paste(drape, (0, dr_y), drape)
    return base.convert("RGB")

# scoring functions (normalize & weight)
def clamp(v, a, b):
    return max(a, min(b, v))

def compute_score_and_pct(orig_metrics, aft_metrics):
    oY,oS,oC,oA,oB = orig_metrics
    aY,aS,aC,aA,aB = aft_metrics
    dY = aY - oY
    dS = aS - oS
    dC = aC - oC
    dA = abs(aA - oA)
    dB = abs(aB - oB)

    # normalized sub-scores in -1..1
    brightness_score = clamp(dY / 5.0, -1.0, 1.0)
    saturation_score = clamp(dS / 5.0, -1.0, 1.0)
    contrast_score = clamp(dC / 3.0, -1.0, 1.0)
    undertone_score = clamp(1.0 - (dA + dB) / 20.0, 0.0, 1.0)  # 0..1 (higher = better)

    # final weighted score - range approximately -1..1
    final = (brightness_score * 0.25 +
             saturation_score * 0.25 +
             contrast_score * 0.20 +
             (undertone_score * 2 - 1) * 0.30)  # convert undertone 0..1 to -1..1 weight

    # percentage and 0-10 score
    pct = clamp((final + 1.0) / 2.0 * 100.0, 0.0, 100.0)
    score10 = round(pct / 10.0, 1)
    return final, pct, score10, (dY, dS, dC, aA - oA, aB - oB)

# UI state
if "preview_idx" not in st.session_state:
    st.session_state.preview_idx = None

st.title("PERSONAL COLOR ANALYSIS")

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

if raw and st.session_state.get("last_img", None) != raw:
    st.session_state.preview_idx = None
    st.session_state.last_img = raw

if img is None:
    st.info("Upload an image to start.")
    st.stop()

# small preview
small = img.resize((int(img.width * 0.40), int(img.height * 0.40)))
st.image(small)

# face detection
cv = pil_to_cv(img)
box = detect_face_box(cv)
if box is None:
    st.warning("Face not detected. Try better lighting & straight face.")
    st.stop()

# draw bounding box and show detected face
preview = cv.copy()
px,py,pw,ph = box
cv2.rectangle(preview, (px,py), (px+pw, py+ph), (0,255,0), 3)
st.image(cv_to_pil(preview), caption="Detected Face", use_container_width=False)

# skin extraction metrics shown (only values)
roi = cv[py:py+ph, px:px+pw]
mask = skin_mask_roi(roi)
Y, Cb, Cr = median_skin(roi, mask)
st.success(f"Skin detected: Y={Y:.1f}  Cb={Cb:.1f}  Cr={Cr:.1f}")

# compute original face metrics once
x,y,w,h = box
face_before = img.crop((x, y, x+w, y+h))
orig_metrics = face_metrics(face_before)

# compute score for all colors automatically
all_scores = []
for idx, row in df.iterrows():
    hx = row["HEX"]
    rgb = hex_to_rgb(hx)
    after_img = build_drape(img, box, rgb)
    face_after = after_img.crop((x, y, x+w, y+h))
    aft_metrics = face_metrics(face_after)
    final, pct, score10, deltas = compute_score_and_pct(orig_metrics, aft_metrics)
    all_scores.append({
        "df_index": idx,
        "ID": int(row["ID"]),
        "ColorName": row["ColorName"],
        "HEX": hx,
        "Category": row.get("Category", "Palette"),
        "final": final,
        "pct": pct,
        "score10": score10,
        "deltas": deltas
    })
scores_df = pd.DataFrame(all_scores)

# show Top 5 Best and Top 5 Worst automatically
top5 = scores_df.sort_values("final", ascending=False).head(5).reset_index(drop=True)
bot5 = scores_df.sort_values("final", ascending=True).head(5).reset_index(drop=True)

# -------------------------
# 2) Best & Worst Colors
# -------------------------

st.header("2) Best & Worst Colors For You")

# ---------- BEST ----------
st.subheader("Top 5 — Best Colors")
best_cols = st.columns(5)

for i, row in top5.iterrows():
    with best_cols[i]:
        st.markdown(
            f"<div style='height:64px;border-radius:8px;background:{row['HEX']}'></div>",
            unsafe_allow_html=True
        )
        st.write(row["ColorName"])
        
        if st.button("Preview", key=f"best_preview_{i}"):
            st.session_state.preview_idx = int(row["df_index"])

# ---------- WORST (Below Best) ----------
st.subheader("Bottom 5 — Worst Colors")
worst_cols = st.columns(5)

for i, row in bot5.iterrows():
    with worst_cols[i]:
        st.markdown(
            f"<div style='height:64px;border-radius:8px;background:{row['HEX']}'></div>",
            unsafe_allow_html=True
        )
        st.write(row["ColorName"])
       
        if st.button("Preview", key=f"worst_preview_{i}"):
            st.session_state.preview_idx = int(row["df_index"])


# Display palette grouped by Category with 6 colors each (as requested)
st.header("3) Color Families — choose any color to preview")
categories = df["Category"].unique().tolist()
# preserve the order of categories: Blues, Purples, Pinks, Oranges, Yellows, Greens, Neutrals if present
preferred_order = ["Blue", "Purple", "Pink", "Orange", "Yellow", "Green", "Neutral", "Palette"]
ordered = [c for c in preferred_order if c in categories] + [c for c in categories if c not in preferred_order]

for cat in ordered:
    with st.expander(cat):
        rows = df[df["Category"] == cat].reset_index()
        cols = st.columns(6)
        for i, r in rows.iterrows():
            hx = r["HEX"]
            name = r["ColorName"]
            idx = int(r["index"])
            col = cols[i % 6]
            with col:
                st.markdown(f"<div style='height:48px;border-radius:6px;background:{hx}'></div>", unsafe_allow_html=True)
                st.write(name)
                # small preview button (no "select" word)
                if st.button("Preview", key=f"preview_{idx}"):
                    st.session_state.preview_idx = idx

# Preview area: show drape + metrics for selected color (if any)
# --------------------------
# 4) PREVIEW & METRICS (FIXED)
# --------------------------
# --------------------------
# 4) PREVIEW & METRICS (MULTIPLE PREVIEWS SAVED)
# --------------------------

st.header("4) Preview & Metrics")

# ------- ALWAYS SHOW ORIGINAL -------
st.subheader("Original Face")
st.image(face_before, width=300)
st.write(f"Brightness: {orig_metrics[0]:.1f}")
st.write(f"Saturation: {orig_metrics[1]:.1f}")
st.write(f"Contrast: {orig_metrics[2]:.1f}")

# Create preview history if not exists
if "preview_history" not in st.session_state:
    st.session_state.preview_history = []

st.markdown("---")

# ------- ADD NEW PREVIEW WHEN BUTTON CLICKED -------
if st.session_state.preview_idx is not None:
    
    sel_idx = st.session_state.preview_idx
    sel = df.loc[sel_idx]

    rgb = hex_to_rgb(sel["HEX"])
    after_img = build_drape(img, box, rgb)
    face_after = after_img.crop((x, y, x+w, y+h))
    aft_metrics = face_metrics(face_after)

    final, pct, score10, deltas = compute_score_and_pct(orig_metrics, aft_metrics)
    dY, dS, dC, dA, dB = deltas

    # save preview to history
    st.session_state.preview_history.append({
        "name": sel["ColorName"],
        "hex": sel["HEX"],
        "img": face_after,
        "metrics": aft_metrics,
        "deltas": deltas,
        "score": score10,
        "pct": pct
    })

    # reset preview idx so it doesn’t trigger again automatically
    st.session_state.preview_idx = None

# ------- SHOW ALL PREVIEWS BELOW ORIGINAL -------
st.subheader("Your Selected Colors (All Previews)")
if len(st.session_state.preview_history) == 0:
    st.info("Click Preview on any color to see how it looks.")
else:
    for p in st.session_state.preview_history:
        st.markdown(f"### {p['name']} — {p['hex']}")
        st.image(p["img"], width=300)

        Y,S,C,A,B = p["metrics"]
        dY,dS,dC,dA,dB = p["deltas"]

        st.write(f"Brightness: {Y:.1f} (Δ {dY:+.1f})")
        st.write(f"Saturation: {S:.1f} (Δ {dS:+.1f})")
        st.write(f"Contrast: {C:.1f}   (Δ {dC:+.1f})")
        st.write(f"LAB A: {A:.2f}      (Δ {dA:+.2f})")
        st.write(f"LAB B: {B:.2f}      (Δ {dB:+.2f})")

        st.write(f"### Score: **{p['score']}/10**")
        if p["pct"] >= 70:
           
            st.write("YES — Highly Suitable")
        elif p["pct"] >= 50:
          
            st.write("OK — Moderately Suitable")
        else:
          
            st.write("NO — Not Suitable")
        st.markdown("---")


# Clear button
if st.button("Clear All Previews"):
    st.session_state.preview_history = []
    st.success("All previews cleared.")


