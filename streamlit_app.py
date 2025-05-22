import streamlit as st
import sqlite3
import os
import uuid
import cv2
from datetime import datetime
from passlib.hash import bcrypt
from urllib.parse import unquote, quote
from urllib.parse import parse_qs, urlparse
from PIL import UnidentifiedImageError
from base64 import b64encode
import base64
import time
import streamlit.components.v1 as components
from collections import defaultdict

import clip
import torch
from PIL import Image


# Load CLIP MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cache - help the program run smootly, avoiding creating new quesry every rerun 
@st.cache_resource
def load_clip_model():
    try:
        model, preprocess = clip.load("ViT-B/32", device="cpu")
        return model, preprocess
    except Exception as e:
        st.error(f"Failed to load CLIP model: {e}")
        return None, None

# Create table for first time use 
def initialize_database(db_path="app.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        hashed_password TEXT NOT NULL,
        role TEXT NOT NULL
    )
    """)
    
    # Create uploads table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        img1_path TEXT NOT NULL,
        assigned_to TEXT,
        timestamp TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

clip_model, clip_preprocess = load_clip_model()

def compare_images_clip(image_path1, image_path2):
    try:
        image1 = clip_preprocess(Image.open(image_path1)).unsqueeze(0).to(device)
        image2 = clip_preprocess(Image.open(image_path2)).unsqueeze(0).to(device)

        with torch.no_grad():
            feat1 = clip_model.encode_image(image1)
            feat2 = clip_model.encode_image(image2)

        similarity = torch.cosine_similarity(feat1, feat2).item()

        # Convert to 0–100% scale
        similarity_percent = round((similarity + 1) / 2 * 100, 2)
        return similarity_percent
    except Exception as e:
        return f"Error comparing with CLIP: {str(e)}"

def get_all_clip_embeddings(images):
    embeddings = {}
    for label, path in images:
        try:
            image_tensor = clip_preprocess(Image.open(path)).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = clip_model.encode_image(image_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embeddings[path] = embedding
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return embeddings

def find_duplicates(embeddings, threshold=0.95):
    duplicates = []
    paths = list(embeddings.keys())
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            sim = torch.cosine_similarity(embeddings[paths[i]], embeddings[paths[j]]).item()
            if sim >= threshold:
                duplicates.append((paths[i], paths[j], round(sim * 100, 2)))
    return duplicates


if os.path.exists(".first_visit_marker"):
    st.session_state["first_visit_done"] = True
    st.session_state["show_welcome"] = False

def get_base64_bg(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_base64 = get_base64_bg("assets/AI.png")


st.markdown(f"""
    <style>
        /* Background container behind everything */
        .background {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-image: url("data:image/png;base64,{bg_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            z-index: -2;
        }}

        /* Overlay dim layer */
        .dim-layer {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: -1;
        }}

        /* Transparent content boxes */
        .stApp {{
            background: transparent;
        }}

        /* Optional: adjust containers if needed */
        .css-1d391kg, .css-ffhzg2, .css-1kyxreq {{
            background-color: rgba(20, 20, 20, 0.7) 
        }}
    </style>

    <!-- Render layers -->
    <div class="background"></div>
    <div class="dim-layer"></div>
""", unsafe_allow_html=True)

# STYLE FOR NAVIGATION BAR 
st.markdown("""
    <style>
        /* Sidebar background & style */
        section[data-testid="stSidebar"] {
            background-color: rgba(30, 30, 30, 0.85);  /* dark semi-transparent */
            backdrop-filter: blur(4px);                /* soft blur effect */
            border-right: 1px solid rgba(255,255,255,0.1);
        }

        /* Sidebar text color */
        section[data-testid="stSidebar"] .css-1v3fvcr {
            color: #ffffff !important;
        }

        /* Highlighted radio button */
        .css-1v3fvcr > div[role="radiogroup"] > label[data-baseweb="radio"] {
            color: #ddd !important;
        }

        /* Selected option appearance */
        .css-1v3fvcr > div[role="radiogroup"] > label[data-baseweb="radio"][aria-checked="true"] {
            background-color: #4CAF50 !important;
            color: black !important;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# === Configuration ===
DB_PATH = "app.db"
RESULT_DIR = "static/results"
os.makedirs(RESULT_DIR, exist_ok=True)


# === Session State Setup ===
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.user = None
    st.session_state.selected = {"img1": None, "img2": None}
    st.session_state.page = "Home"
    st.session_state.first_visit_done = False
    st.session_state.show_welcome = True
else:
    # Ensure all keys exist without resetting them
    st.session_state.setdefault("user", None)
    st.session_state.setdefault("selected", {"img1": None, "img2": None})
    st.session_state.setdefault("page", "Home")
    st.session_state.setdefault("first_visit_done", False)
    st.session_state.setdefault("show_welcome", not st.session_state.first_visit_done)
    
# === DB Connection ===
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# === WELCOME PAGE ===
def render_welcome_page():
    st.title("👋 Welcome to the AI Image Comparison App")
    st.markdown("""
    This app lets you:
    - Upload shelf or product images.
    - Compare images using the ORB algorithm.
    - Compare images using the CLIP Model (AI).
    - Manage uploads and user access by roles (Guest, Uploader, Admin).

    ### How to Use:
    1. Click **Login** to sign in or register.
    2. Navigate using the **sidebar**.
    3. Upload or assign images.
    4. Compare and analyze results.

    Admin code will be: **IamAdmin**
        
    Created by: **Nhat Vo**
    
    Contact: nhatvm30012004@gmail.com

    THANKS FOR USING MY APP
    ---
    """, unsafe_allow_html=True)

    if st.button("👉 Continue to Login"):
        st.session_state.first_visit_done = True
        st.session_state.show_welcome = False
        with open(".first_visit_marker", "w") as f:
            f.write("done")
        st.rerun()
    
# === INSTRUCTION PAGE ===
def render_instruction_page():
    st.markdown("# 👋 Welcome to the AI Image Comparison App")
    st.markdown("""
    This web app allows you to:
    
    ### 🧭 Navigation Guide:
    - **Home**: Upload / Choose from current Library and compare your own images.
    - **ORB Method**: Learn how the ORB algorithm works.
    - **Guest View / Admin Dashboard** (Admin only): View uploads and manage images.
    - **Manage Users** (Admin only): Change user roles or delete users.
    
    ### 📸 How to Compare Images:
    1. Go to **Home**.
    2. Upload one or two images.
    3. Click `🔍 Compare by ORB` to see the similarity result.
    4. Or click `🔍 Compare by CLIP` to see the similarity result.
    5. You are able to compare between 2 methods 

    ### 🔐 Roles Overview:
    - **Guest**: Upload and compare your own assigned images.
    - **Uploader**: Upload images and assign them to guests.
    - **Admin**: Full access to manage users, see all images, and assign roles.
    
    ### ✅ Tips:
    - All images are stored securely.
    - You can delete any of your uploads.
    - Use **Home** button on the top right to return at any time.
    
    ---
    """, unsafe_allow_html=True)

    st.success("You're now ready to explore the app! Use the sidebar to begin.")

# === RENDERING CLIP PAGE ===
def render_clip_method_page():
    st.markdown("# 🤖 CLIP Method Explanation")

    st.markdown("""
    **CLIP (Contrastive Language–Image Pre-Training)** is a powerful model developed by OpenAI that understands images and text together.

    ### 🧠 How it works:

    1. **Joint Vision & Language Training**  
       CLIP is trained on 400 million (image, text) pairs. It learns to **embed images and texts into the same space**.

    2. **Encoding Images and Text**  
       - Images go through a Vision Transformer (ViT).
       - Text goes through a transformer-based language encoder.
       - Both outputs are compared using **cosine similarity**.

    3. **Zero-shot Recognition**  
       Once trained, CLIP can recognize new image concepts **without retraining** — just by comparing the image to text labels like:  
       `"a photo of a cat"`, `"a photo of a dog"`...

    ### 🧪 What we're using it for:

    In this app, we use CLIP to compare two images. If CLIP's cosine similarity score is close to 1, the images are visually and semantically similar.

    ### 📏 Similarity Score:

    - **100%** → Very similar
    - **50%** → Some similarity
    - **0–30%** → Likely unrelated
    """, unsafe_allow_html=True)

# === Image Comparison by ORB ===
def compare_images(img1_path, img2_path, result_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        raise ValueError("Images could not be loaded.")
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        result = cv2.hconcat([img1, img2])
        similarity = 0.0
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None)
        similarity = round(100 * len(matches) / max(len(kp1), len(kp2)), 2)
    cv2.imwrite(result_path, result)
    return similarity

# def get_clip_embedding(img_path):
#     img = Image.open(img_path).convert("RGB")
#     img_tensor = clip_preprocess(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         features = clip_model.encode_image(img_tensor)
#         features = features / features.norm(dim=-1, keepdim=True)
#     return features

# def compare_clip_images(img1_path, img2_path):
#     feat1 = get_clip_embedding(img1_path)
#     feat2 = get_clip_embedding(img2_path)
#     similarity = (feat1 @ feat2.T).item()
#     return round(similarity * 100, 2)


# === Remove DB records pointing to missing files ===
def clean_missing_uploads():
    db = get_db()
    uploads = db.execute("SELECT id, img1_path FROM uploads").fetchall()
    removed = []

    for row in uploads:
        path = row["img1_path"].lstrip("/")
        if not os.path.exists(path):
            db.execute("DELETE FROM uploads WHERE id = ?", (row["id"],))
            removed.append(path)
        else:
            try:
                from PIL import Image
                Image.open(path)
            except Exception:
                db.execute("DELETE FROM uploads WHERE id = ?", (row["id"],))
                try:
                    os.remove(path)
                except: pass
                removed.append(path)

    db.commit()
    db.close()

    if removed:
        print("🧹 Removed missing or corrupt files:")
        for r in removed:
            print(f"  - {r}")

# === Login ===
def login_ui():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        db.close()
        if user and bcrypt.verify(password, user["hashed_password"]):
            st.session_state.user = {"username": user["username"], "role": user["role"]}
            # Manually reload with URL query params for refresh-proofing
            st.markdown(
                f'<meta http-equiv="refresh" content="0;URL=/?user={quote(user["username"])}&role={quote(user["role"])}">',
                unsafe_allow_html=True,
            )
            st.stop()
        else:
            st.error("Invalid credentials")
    st.markdown("Don't have an account? [Register here](?register=1)")
            
# === Register Dashboard ===           
def register_ui():
    st.title("📝 Register")

    username = st.text_input("Choose a username")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    role = st.selectbox("Select Role", ["guest", "uploader", "admin"])

    # Only show secret code input if 'admin' is selected
    admin_code = None
    if role == "admin":
        admin_code = st.text_input("Admin Secret Code", type="password")

    if st.button("Register"):
        if password != confirm:
            st.error("❌ Passwords do not match.")
            return

        if role == "admin" and admin_code != "IamAdmin":
            st.error("❌ Invalid admin secret code.")
            return

        db = get_db()
        existing = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            st.error("❌ Username already exists.")
            db.close()
            return

        hashed_pw = bcrypt.hash(password)
        db.execute(
            "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
            (username, hashed_pw, role)
        )
        db.commit()
        db.close()
        st.success("✅ Account created. You can now login.")
        st.markdown('<meta http-equiv="refresh" content="0; URL=/" />', unsafe_allow_html=True)
        st.stop()

# HIGLIGHTING AND DIMMING THE PICTURE 
def display_image_base64(path, label, selected):
    try:
        with open(path, "rb") as img_file:
            img_bytes = img_file.read()
            encoded = b64encode(img_bytes).decode()

        style = ""
        border = "none"

        if selected:
            style = "filter: brightness(1.2); border: 3px solid lightgreen; border-radius: 8px;"
        else:
            # Check if this image belongs to any duplicate group
            color_list = ["#2196F3", "#E91E63", "#FF9800", "#4CAF50", "#9C27B0", "#795548"]
            duplicate_groups = st.session_state.get("duplicate_groups", [])
            for i, group in enumerate(duplicate_groups):
                if path in group:
                    border = f"3px solid {color_list[i % len(color_list)]}"
                    break
            style = f"filter: brightness(0.6); border: {border}; border-radius: 8px;"

        return f"""
        <div style="text-align:center;">
            <img src="data:image/jpeg;base64,{encoded}" width="150" style="{style}"/>
            <div style="margin-top:4px;">{label}</div>
        </div>
        """
    except Exception:
        return f"<div style='color:red;'>❌ Error loading: {path}</div>"

# === Guest Dashboard ===
def guest_dashboard():
    clean_missing_uploads() 
    st.title(f"👤 {st.session_state.user['username']} Dashboard")
    clean_missing_uploads()

    db = get_db()
    uploads = db.execute(
        "SELECT * FROM uploads WHERE assigned_to = ?",
        (st.session_state.user["username"],)
    ).fetchall()
    db.close()

    available_images = [
        (f"Image {i+1}", row["img1_path"].lstrip("/"))
        for i, row in enumerate(uploads)
        if os.path.exists(row["img1_path"].lstrip("/"))
    ]

    st.markdown("## 🆚 Compare Images")
    col1, col2 = st.columns(2)
    upload_paths = {}

    # --- IMAGE 1 ---
    with col1:
        st.markdown("#### Image 1")
        use_upload1 = st.toggle("Upload from computer", key="upload1_toggle")
        img1_path = None
        if use_upload1:
            img1_file = st.file_uploader("Upload", key="img1_file")
            if img1_file:
                temp = os.path.join(RESULT_DIR, f"temp1_{uuid.uuid4().hex}.jpg")
                with open(temp, "wb") as f:
                    f.write(img1_file.read())
                img1_path = temp
                upload_paths["img1"] = (img1_file, temp)
        else:
            label1 = st.selectbox("Choose from uploaded", [""] + [l for l, _ in available_images], key="img1_select")
            img1_path = dict(available_images).get(label1) if label1 else None
        st.session_state.selected["img1"] = img1_path

    # --- IMAGE 2 ---
    with col2:
        st.markdown("#### Image 2")
        use_upload2 = st.toggle("Upload from computer", key="upload2_toggle")
        img2_path = None
        if use_upload2:
            img2_file = st.file_uploader("Upload", key="img2_file")
            if img2_file:
                temp = os.path.join(RESULT_DIR, f"temp2_{uuid.uuid4().hex}.jpg")
                with open(temp, "wb") as f:
                    f.write(img2_file.read())
                img2_path = temp
                upload_paths["img2"] = (img2_file, temp)
        else:
            label2 = st.selectbox("Choose from uploaded", [""] + [l for l, _ in available_images], key="img2_select")
            img2_path = dict(available_images).get(label2) if label2 else None
        st.session_state.selected["img2"] = img2_path
    
    # Upload to DB
    if st.button("📤 Upload"):
        db = get_db()
        for key, (file_obj, temp_path) in upload_paths.items():
            filename = f"{st.session_state.user['username']}_{key}_{uuid.uuid4().hex}.jpg"
            final_path = os.path.join(RESULT_DIR, filename)
            os.rename(temp_path, final_path)
            db.execute("INSERT INTO uploads (username, img1_path, assigned_to, timestamp) VALUES (?, ?, ?, ?)",
                    (st.session_state.user["username"], f"/static/results/{filename}", st.session_state.user["username"], datetime.utcnow()))
        db.commit()
        db.close()

        # Clear all related selections (but skip modifying widget keys)
        st.session_state.selected = {"img1": None, "img2": None}
        st.session_state["duplicate_groups"] = []

        # Do not reset toggles here — Streamlit doesn't allow that
        st.success("✅ Uploaded successfully.")
        st.rerun()
        
        # Reset selections or toggles if needed
        st.session_state.selected = {"img1": None, "img2": None}
        if "upload1_toggle" in st.session_state:
            st.session_state.update({"upload1_toggle": False})
        st.session_state["upload2_toggle"] = False

        st.success("✅ Uploaded successfully.")
        st.rerun()  # ✅ soft reload without logging out

    compare_col1, compare_col2 = st.columns([1, 1])
    
    # Compare image by ORB 
    with compare_col1:
        if st.button("🔍 Compare by ORB"):
            if img1_path and img2_path:
                try:
                    result_path = os.path.join(RESULT_DIR, f"{uuid.uuid4()}_result.jpg")
                    sim = compare_images(img1_path, img2_path, result_path)
                    st.success(f"ORB Similarity: {sim}%")
                    st.image(result_path, caption="🖼️ ORB Comparison Result")
                except Exception as e:
                    st.error(f"❌ ORB Comparison failed: {str(e)}")
            else:
                st.warning("⚠️ Provide both images.")

    # Compare image by CLIP 
    with compare_col2:
        if st.button("🤖 Compare by CLIP"):
            if img1_path and img2_path:
                clip_score = compare_images_clip(img1_path, img2_path)
                if isinstance(clip_score, str):
                    st.error(clip_score)
                else:
                    st.success(f"CLIP Cosine Similarity: {clip_score:.2f}%")
            else:
                st.warning("⚠️ Provide both images.")
    
    
           
        components.html(f"""
        <style>
        .home-floating {{
            position: fixed;
            bottom: 25px;
            right: 25px;
            background-color: #292929;
            padding: 10px 20px;
            border-radius: 10px;
            border: 1px solid #4CAF50;
            color: white;
            font-weight: bold;
            cursor: pointer;
            z-index: 9999;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            transition: all 0.2s ease-in-out;
        }}
        .home-floating:hover {{
            background-color: #4CAF50;
            color: black;
            transform: scale(1.03);
        }}
        </style>
        <div class="home-floating" onclick="window.parent.postMessage({{ type: 'go_home' }}, '*')">🏠 Home Page</div>
    """, height=100)

        components.html("""
        <script>
            window.addEventListener("message", function(event) {
                if (event.data.type === "go_home") {
                    const streamlitEvents = window.parent.streamlitEvents;
                    if (streamlitEvents) {
                        streamlitEvents.send({
                            type: "streamlit:setComponentValue",
                            key: "page",
                            value: "Home"
                        });
                    }
                }
            });
        </script>
    """, height=0)

    # ---- GALLERY WITH LIVE HIGHLIGHTING ----
    title_col, button_col = st.columns([5, 2])  # Adjust width ratio

    with title_col:
        st.markdown(f"### 📸 Uploaded Images ({len(available_images)})")

    if st.button("🧬 Detect Duplicates"):
        embeddings = get_all_clip_embeddings(available_images)
        dups = find_duplicates(embeddings, threshold=0.95)

        if dups:
            st.session_state["duplicate_groups"] = []
            seen = set()

            for path1, path2, _ in dups:
                if path1 in seen and path2 in seen:
                    continue
                found = False
                for group in st.session_state["duplicate_groups"]:
                    if path1 in group or path2 in group:
                        group.update([path1, path2])
                        found = True
                        break
                if not found:
                    st.session_state["duplicate_groups"].append(set([path1, path2]))
                seen.update([path1, path2])

            st.success(f"🔍 Found {len(st.session_state['duplicate_groups'])} duplicate group(s). Look at the color borders below.")
        else:
            st.session_state["duplicate_groups"] = []
            st.success("🎉 No duplicates found!")
    
    
    cols_per_row = 4
    for row_start in range(0, len(available_images), cols_per_row):
        row_images = available_images[row_start:row_start + cols_per_row]
        row_cols = st.columns(len(row_images))  # Only as many columns as needed

        for col, (label, path) in zip(row_cols, row_images):
            with col:
                selected_img1 = st.session_state.selected.get("img1")
                selected_img2 = st.session_state.selected.get("img2")
                is_selected = (path == selected_img1) or (path == selected_img2)

                st.markdown(display_image_base64(path, label, is_selected), unsafe_allow_html=True)

                delete_key = f"delete_{row_start}_{label}"
                if st.button("🗑️ Delete", key=delete_key):
                    db = get_db()
                    db.execute("DELETE FROM uploads WHERE username = ? AND img1_path = ?", (
                        st.session_state.user["username"],
                        f"/{path}" if not path.startswith("/") else path
                    ))
                    db.commit()
                    db.close()
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    st.success(f"Deleted {label}")
                    st.rerun()

def admin_dashboard():
    clean_missing_uploads() 
    st.title(f"🛠️ Admin Dashboard (Full Access)")

    clean_missing_uploads()
    db = get_db()
    uploads = db.execute("SELECT * FROM uploads").fetchall()
    db.close()

    # Build list of all images
    available_images = [(f"{row['username']} - Image {i+1}", row["img1_path"].lstrip("/"))
                        for i, row in enumerate(uploads)
                        if os.path.exists(row["img1_path"].lstrip("/"))]

    st.markdown("## 🆚 Compare Any Images")
    col1, col2 = st.columns(2)
    img1_path, img2_path = None, None

    with col1:
        img1_label = st.selectbox("Select Image 1", [label for label, _ in available_images], key="admin_img1")
        img1_path = dict(available_images).get(img1_label)

    with col2:
        img2_label = st.selectbox("Select Image 2", [label for label, _ in available_images], key="admin_img2")
        img2_path = dict(available_images).get(img2_label)

    if st.button("🔍 Compare Selected"):
        if img1_path and img2_path:
            result_path = os.path.join(RESULT_DIR, f"{uuid.uuid4()}_admin_result.jpg")
            try:
                sim = compare_images(img1_path, img2_path, result_path)
                st.success(f"Similarity: {sim}%")
                st.image(result_path, caption="🖼️ Comparison Result")
            except Exception as e:
                st.error(f"Comparison failed: {e}")
        else:
            st.warning("⚠️ Select two valid images to compare.")

    st.markdown("## 📸 All Uploaded Images (Manageable)")
    cols = st.columns(4)
    db = get_db()
    for i, (label, path) in enumerate(available_images):
        with cols[i % 4]:
            st.image(path, width=150, caption=label)
            if st.button("🗑️ Delete", key=f"delete_{i}"):
                db.execute("DELETE FROM uploads WHERE img1_path = ?", ("/" + path,))
                if os.path.exists(path):
                    os.remove(path)
                db.commit()
                db.close()
                st.success(f"Deleted {label}")
                st.markdown('<meta http-equiv="refresh" content="0; URL=/" />', unsafe_allow_html=True)
                st.stop()
                
    # UPLOAD IMAGE AS AN ADMIN 
    st.markdown("## 📤 Upload Image as Admin")
    admin_img = st.file_uploader("Upload image")

    if admin_img:
        filename = f"{st.session_state.user['username']}_admin_{uuid.uuid4().hex}.jpg"
        path = os.path.join(RESULT_DIR, filename)

        # Save file to disk
        with open(path, "wb") as f:
            f.write(admin_img.read())

        # Save record to DB
        db = get_db()
        db.execute(
            "INSERT INTO uploads (username, img1_path, assigned_to, timestamp) VALUES (?, ?, ?, ?)",
            (st.session_state.user["username"], f"/static/results/{filename}", None, datetime.utcnow())
        )
        db.commit()
        db.close()
        st.success("✅ Uploaded successfully.")
        # Clear the upload toggles and selection
        st.session_state.selected = {"img1": None, "img2": None}
        st.rerun()

def uploader_dashboard():
    clean_missing_uploads() 
    st.title(f"📦 Uploader: {st.session_state.user['username']}")
    clean_missing_uploads()

    db = get_db()
    guests = db.execute("SELECT username FROM users WHERE role = 'guest'").fetchall()
    guest_list = [g["username"] for g in guests]

    st.subheader("📤 Upload and Assign")
    assigned_to = st.selectbox("Assign to", guest_list)
    image = st.file_uploader("Upload Image", key="uploader_image")

    if image and assigned_to:
        filename = f"{st.session_state.user['username']}_upload_{uuid.uuid4().hex}.jpg"
        path = os.path.join(RESULT_DIR, filename)
        with open(path, "wb") as f:
            f.write(image.read())

        db.execute("INSERT INTO uploads (username, img1_path, assigned_to, timestamp) VALUES (?, ?, ?, ?)",
                   (st.session_state.user["username"], f"/static/results/{filename}", assigned_to, datetime.utcnow()))
        db.commit()
        st.success("✅ Image uploaded and assigned.")
        st.markdown('<meta http-equiv="refresh" content="0; URL=/" />', unsafe_allow_html=True)
        st.stop()

    uploads = db.execute("SELECT * FROM uploads WHERE username = ?", (st.session_state.user["username"],)).fetchall()
    db.close()

    st.subheader("📋 Your Uploads")
    cols = st.columns(4)
    for i, row in enumerate(uploads):
        path = row["img1_path"].lstrip("/")
        if os.path.exists(path):
            with cols[i % 4]:
                st.image(path, width=150, caption=f"→ {row['assigned_to']}")
                delete_key = f"delete_uploader_{i}"
                if st.button("🗑️ Delete", key=delete_key):
                    db = get_db()
                    db.execute("DELETE FROM uploads WHERE username = ? AND img1_path = ?", (
                        st.session_state.user["username"],
                        f"/{path}" if not path.startswith("/") else path
                    ))
                    db.commit()
                    db.close()
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    st.success(f"Deleted image {i+1}")
                    st.markdown('<meta http-equiv="refresh" content="0; URL=/" />', unsafe_allow_html=True)
                    st.stop()
                    
def admin_manage_users():
    st.title("👥 Manage Users")

    db = get_db()
    users = db.execute("SELECT * FROM users").fetchall()

    for user in users:
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

        with col1:
            st.markdown(f"**{user['username']}**")

        with col2:
            new_role = st.selectbox(
                "Role",
                ["guest", "uploader", "admin"],
                index=["guest", "uploader", "admin"].index(user["role"]),
                key=f"role_{user['id']}"
            )

        with col3:
            if st.button("Update", key=f"update_{user['id']}"):
                db.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user["id"]))
                db.commit()
                st.success(f"✅ Updated role for {user['username']} to {new_role}")

        with col4:
            if st.button("🗑️ Delete", key=f"delete_{user['id']}"):
                db.execute("DELETE FROM users WHERE id = ?", (user["id"],))
                db.commit()
                st.success(f"🧨 Deleted user {user['username']}")
                st.markdown('<meta http-equiv="refresh" content="0; URL=/" />', unsafe_allow_html=True)
                st.stop()
    db.close()

# === RENDERING HOME PAGE BUTTON ===
def top_bar_controls():
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🚪 Logout", key="logout_button"):
                st.session_state.clear()
                st.markdown('<meta http-equiv="refresh" content="0; URL=/" />', unsafe_allow_html=True)
                st.stop()
        with col2:
            if st.session_state.get("page") != "Home":
                if st.button("🏠 Home", key="home_button"):
                    st.session_state.page = "Home"
                    st.session_state["freeze_radio"] = True
                    st.session_state["nav_key"] = str(uuid.uuid4()) 
                    st.rerun()
                
# === Logout UI ===
def logout_ui():
    top_bar_controls()
    
# === Query param re-auth (on reload) ===

query_string = st.query_params

# Registration page routing
if "register" in query_string:
    register_ui()
    st.stop()

# Restore session if query has login info
if "user" in query_string and "role" in query_string and st.session_state.user is None:
    st.session_state.user = {
        "username": unquote(query_string["user"]),
        "role": unquote(query_string["role"]),
    }

# If malformed or expired session: force to login cleanly
if ("user" in query_string or "role" in query_string) and st.session_state.user is None:
    st.markdown('<meta http-equiv="refresh" content="0; URL=/" />', unsafe_allow_html=True)
    st.stop()


# === MAIN ===
if st.session_state.show_welcome and not st.session_state.user:
    print(st.session_state.user)
    render_welcome_page()

elif not st.session_state.user:
    login_ui()

else:
    clean_missing_uploads()

    # Show Logout + Home in one row
    top_bar_controls()

    # Ensure default page is set
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    nav_key = st.session_state.get("nav_key", "nav_default")  
    
    role = st.session_state.user["role"]
    radio_frozen = st.session_state.get("freeze_radio", False)

    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    if role == "admin":
        nav_choice = st.sidebar.radio(
            "Choose Page",
            ["Instruction", "Home", "Guest View", "Admin Dashboard", "Manage Users", "ORB Method", "CLIP Model"],
            index=["Instruction", "Home", "Guest View", "Admin Dashboard", "Manage Users", "ORB Method", "CLIP Model"].index(st.session_state.page),
            key=nav_key
        )
    else:
        nav_choice = st.sidebar.radio(
            "Choose Page",
            ["Instruction", "Home", "ORB Method", "CLIP Model"],
            index=["Instruction", "Home", "ORB Method", "CLIP Model"].index(st.session_state.page),
            key=nav_key
        )

    # Only update page if not frozen by Home button
    if not radio_frozen:
        if st.session_state.page != nav_choice:
            st.session_state.page = nav_choice
            st.rerun()  # 🔁 force UI to re-render immediately

    # Clean up after freezing (if any)
    st.session_state.pop("freeze_radio", None)
    st.session_state.pop("nav_key", None)

    # --- Render Page Content ---
    if st.session_state.page == "Home":
        if role == "guest":
            guest_dashboard()
        elif role == "uploader":
            uploader_dashboard()
        elif role == "admin":
            guest_dashboard()
    elif st.session_state.page == "Instruction":
        render_instruction_page()
    
    elif st.session_state.page == "Guest View":
        guest_dashboard()

    elif st.session_state.page == "Admin Dashboard":
        admin_dashboard()

    elif st.session_state.page == "Manage Users":
        admin_manage_users()

    elif st.session_state.page == "ORB Method":
        st.markdown("## 🧠 ORB Method Explanation")
       
        st.markdown("""
        <div style="
            background-color: rgba(255, 0, 0, 0.15); 
            padding: 1rem 1.2rem;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            text-align: center;
            margin-bottom: 1rem;
            border: 1px solid rgba(255, 0, 0, 0.3);
        ">
            🚨 <strong>NOTE:</strong> This method is still unstable and may need trained data for accuracy.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
                    
        🔍 Here's how it works step-by-step:

        1. **FAST Keypoint Detection**  
        ORB uses FAST to find corner-like features in an image.

        2. **Orientation Assignment**  
        ORB computes the direction of intensity change around each keypoint — making it rotation-invariant.

        3. **Harris Corner Ranking**  
        It keeps only the strongest keypoints based on a corner strength measure.

        4. **BRIEF Descriptor**  
        A binary vector that describes the keypoint’s neighborhood.

        5. **Rotation-aware BRIEF**  
        ORB modifies BRIEF to account for orientation so it can match rotated images.

        In this app, ORB helps detect and match features between two shelf images even if they’re rotated or angled.
        """, unsafe_allow_html=True)
        
    elif st.session_state.page == "CLIP Model":
        render_clip_method_page()