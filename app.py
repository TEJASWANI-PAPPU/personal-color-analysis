import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import io, os

st.set_page_config(page_title="Personal Color Analysis", layout="wide")

CSV_NAME = "skin_tone_color_analysis_dataset.csv"
df = pd.read_csv(CSV_NAME)

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
    ycc=cv2.cvtColor(roi,cv2.COLOR_BGR2YCrCb)
    Y,Cr,Cb = ycc[:,:,0], ycc[:,:,1], ycc[:,:,2]
    mask = ((Cb>=77)&(Cb<=127)&(Cr>=133)&(Cr<=173)).astype(np.uint8)*255
    kernel=np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    return mask

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

def face_metrics(face):
    arr=np.array(face)
    if arr.size==0:
        return 0,0,0,0,0
    y=cv2.cvtColor(arr,cv2.COLOR_RGB2YCrCb)[:,:,0].mean()
    s=cv2.cvtColor(arr,cv2.COLOR_RGB2HSV)[:,:,1].mean()
    c=arr.std()
    lab=cv2.cvtColor(arr,cv2.COLOR_RGB2LAB)
    A=lab[:,:,1].mean()
    B=lab[:,:,2].mean()
    return float(y),float(s),float(c),float(A),float(B)

def build_drape(img,box,rgb):
    W,H=img.size
    x,y,w,h = box
    dr_y = y+h - int(H*0.05)
    dr_y = max(0,dr_y)
    dr_h = H-dr_y
    base=img.copy().convert("RGBA")
    drape=Image.new("RGBA",(W,dr_h),(rgb[0],rgb[1],rgb[2],255))
    base.paste(drape,(0,dr_y),drape)
    return base.convert("RGB")

def score_color(orig, aft, is_best, is_worst):
    if is_best:
        return ("YES — This is your BEST color 💖","Always flattering",["Glow","Clear skin"])
    if is_worst:
        return ("NO — This is your WORST color ❌","Not suitable",["Dull","Shadows"])
    oB,oS,oC,oA,oBb = orig
    B,S,C,A,Bb = aft
    dB=B-oB; dS=S-oS; dC=C-oC; dA=A-oA; dBl=Bb-oBb
    score=0
    if dB>1: score+=2
    if dS>1: score+=2
    if dC>0.8: score+=1
    if abs(dA)<5 and abs(dBl)<5: score+=1
    if score>=5:
        return ("YES — suits you","Bright & Vibrant",["Glow","Sharper"])
    elif score>=3:
        return ("OK — Moderately Good","Soft Effect",["Subtle","Natural"])
    else:
        return ("NO — Not Suitable","Dulls Features",["Flat","Low contrast"])

st.title("Personal Color Analysis — PCA System")

if "selected" not in st.session_state:
    st.session_state.selected=[]
if "last_img" not in st.session_state:
    st.session_state.last_img=None

st.header("1) Upload or Take Photo")
cam=st.camera_input("Take Photo")
up=st.file_uploader("Or Upload Image")

img=None; raw=None
if cam:
    raw=cam.getvalue()
    img=Image.open(io.BytesIO(raw)).convert("RGB")
elif up:
    raw=up.getvalue()
    img=Image.open(io.BytesIO(raw)).convert("RGB")

if raw and st.session_state.last_img!=raw:
    st.session_state.selected=[]
    st.session_state.last_img=raw

if img is None: st.stop()

small=img.resize((int(img.width*0.40), int(img.height*0.40)))
st.image(small)

cv=pil_to_cv(img)
box=detect_face_box(cv)
if box is None:
    st.warning("Face not detected.")
    st.stop()

x,y,w,h=box
roi=cv[y:y+h, x:x+w]

mask=skin_mask_roi(roi)
Y,Cb,Cr=median_skin(roi,mask)

Y=int(min(255, Y*1.15))

st.success(f"Skin tone detected: Y={Y} Cb={Cb} Cr={Cr}")

ID, matched_rows = nearest_skin(Y,Cb,Cr,df)
best=matched_rows[matched_rows.Label=="Best"].head(5)
worst=matched_rows[matched_rows.Label=="Worst"].head(5)

st.header("2) Suggested Colors (Best & Worst)")
st.subheader("Best 5 Colors")
best_cols=st.columns(5)

for i in range(5):
    if i < len(best):
        r=best.iloc[i]
        hx=r.HEX
        with best_cols[i]:
            st.markdown(f"<div style='height:60px;border-radius:8px;background:{hx}'></div>",unsafe_allow_html=True)
            if st.button(f"Select Best {i+1}"):
                st.session_state.selected.append({"idx": int(r.name),"type":"best"})

st.subheader("Worst 5 Colors")
worst_cols=st.columns(5)
for i in range(5):
    if i < len(worst):
        r=worst.iloc[i]
        hx=r.HEX
        with worst_cols[i]:
            st.markdown(f"<div style='height:60px;border-radius:8px;background:{hx}'></div>",unsafe_allow_html=True)
            if st.button(f"Select Worst {i+1}"):
                st.session_state.selected.append({"idx": int(r.name),"type":"worst"})

st.header("4) Color Preview — Side by Side Comparison")
face_before=img.crop((x,y,x+w,y+h))
orig_metrics=face_metrics(face_before)

if len(st.session_state.selected)==0:
    st.info("Select any color above to preview.")
else:
    for sel in st.session_state.selected:
        row=df.loc[sel["idx"]]
        rgb=rgb_from_ycbcr(row.Color_Y, row.Color_Cb, row.Color_Cr)
        hx=row.HEX
        after_img=build_drape(img,(x,y,w,h),rgb)
        face_after=after_img.crop((x,y,x+w,y+h))
        aft_metrics=face_metrics(face_after)
        is_best = sel["type"]=="best"
        is_worst = sel["type"]=="worst"
        verdict, subtitle, effects = score_color(orig_metrics, aft_metrics, is_best, is_worst)

        st.markdown(f"## {sel['type'].upper()} — {hx}")
        c1,c2=st.columns([1,1])

        with c1:
            st.write("### Original")
            st.image(small)
            st.write(f"Brightness: {orig_metrics[0]:.1f}")
            st.write(f"Saturation: {orig_metrics[1]:.1f}")
            st.write(f"Contrast: {orig_metrics[2]:.1f}")

        with c2:
            st.write("### With Selected Color")
            st.image(after_img.resize((int(img.width*0.40), int(img.height*0.40))))
            st.write(f"Brightness Δ {aft_metrics[0]-orig_metrics[0]:+.1f}")
            st.write(f"Saturation Δ {aft_metrics[1]-orig_metrics[1]:+.1f}")
            st.write(f"Contrast Δ {aft_metrics[2]-orig_metrics[2]:+.1f}")
            st.markdown(f"### Verdict: **{verdict}**")
            st.markdown(f"**{subtitle}**")
            st.write("Effects: "+", ".join(effects))

      







