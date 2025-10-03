import os
import datetime
from io import BytesIO
import streamlit as st
from PIL import Image

# ---------------- GOOGLE VERTEX (Gemini for Text Refinement) ----------------
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account

PROJECT_ID = st.secrets["gcp_service_account"]["project_id"]
credentials = service_account.Credentials.from_service_account_info(
    dict(st.secrets["gcp_service_account"])
)
vertexai.init(project=PROJECT_ID, location="global", credentials=credentials)

TEXT_MODEL = GenerativeModel("gemini-2.0-flash")  # prompt refinement

# ---------------- HUGGING FACE (DeepFloyd IF / SDXL for Images) ----------------
import torch
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline

# ⚠️ Hardcode your Hugging Face token here
HF_TOKEN = "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"

@st.cache_resource
def load_if_pipeline():
    try:
        pipe = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-M-v1.0",
            token=HF_TOKEN,   # ✅ new way to pass token
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        st.warning(f"⚠️ Could not load DeepFloyd IF, falling back to SDXL. Error: {e}")
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_if_img2img_pipeline():
    try:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "DeepFloyd/IF-I-M-v1.0",
            token=HF_TOKEN,   # ✅ updated
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return pipe.to("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_PIPELINE = load_if_pipeline()
IMAGE_EDIT_PIPELINE = load_if_img2img_pipeline()

# ---------------- HELPERS ----------------
def safe_get_enhanced_text(resp):
    if hasattr(resp, "text") and resp.text:
        return resp.text
    if hasattr(resp, "candidates") and resp.candidates:
        try:
            return resp.candidates[0].content.parts[0].text
        except Exception:
            pass
    return str(resp)

def run_if_generate(prompt, steps=30, guidance=7.5, seed=None):
    generator = torch.Generator("cuda" if torch.cuda.is_available() else "cpu")
    if seed:
        generator.manual_seed(seed)
    out = IMAGE_PIPELINE(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=512, width=512,
        generator=generator
    )
    img = out.images[0]
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def run_if_edit(prompt, base_bytes, strength=0.7, guidance=7.5):
    base_img = Image.open(BytesIO(base_bytes)).convert("RGB")
    out = IMAGE_EDIT_PIPELINE(
        prompt=prompt,
        image=base_img,
        strength=strength,
        guidance_scale=guidance,
    )
    img = out.images[0]
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="DeepFloyd IF + Gemini", layout="wide")
st.title("🖼️ DeepFloyd IF (Images) + Gemini (Prompt Refinement)")

if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
if "edited_images" not in st.session_state:
    st.session_state.edited_images = []

tab_gen, tab_edit = st.tabs(["✨ Generate Images", "🖌️ Edit Images"])

# ---------------- GENERATE ----------------
with tab_gen:
    st.subheader("✨ Generate Images")
    raw_prompt = st.text_area("Enter your idea", height=100)
    num_images = st.slider("Number of images", 1, 3, 1)
    steps = st.slider("Inference steps", 10, 50, 30)
    guidance = st.slider("Guidance scale", 1.0, 15.0, 7.5)

    if st.button("🚀 Generate"):
        if not raw_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Refining with Gemini..."):
                text_resp = TEXT_MODEL.generate_content(raw_prompt)
                enhanced_prompt = safe_get_enhanced_text(text_resp).strip()
                st.info(f"🔮 Enhanced Prompt:\n\n{enhanced_prompt}")

            with st.spinner("Generating with DeepFloyd IF / SDXL..."):
                generated = []
                for i in range(num_images):
                    img_bytes = run_if_generate(enhanced_prompt, steps, guidance)
                    generated.append(img_bytes)

                for idx, img_bytes in enumerate(generated):
                    fname = f"gen_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png"
                    st.session_state.generated_images.append({"filename": fname, "content": img_bytes})
                    st.image(Image.open(BytesIO(img_bytes)), caption=fname, use_container_width=True)
                    st.download_button("⬇️ Download", data=img_bytes, file_name=fname, mime="image/png", key=f"dl_gen_{idx}")

# ---------------- EDIT ----------------
with tab_edit:
    st.subheader("🖌️ Edit Images")
    uploaded_file = st.file_uploader("📤 Upload an image", type=["png","jpg","jpeg"])
    edit_prompt = st.text_area("Enter edit instruction", height=100)
    num_edits = st.slider("Number of edited versions", 1, 3, 1)

    if st.button("🚀 Edit"):
        if not uploaded_file or not edit_prompt.strip():
            st.warning("Upload an image + enter instruction.")
        else:
            base_bytes = uploaded_file.read()
            with st.spinner("Refining instruction with Gemini..."):
                text_resp = TEXT_MODEL.generate_content(edit_prompt)
                enhanced_edit = safe_get_enhanced_text(text_resp).strip()
                st.info(f"🔮 Enhanced Edit Instruction:\n\n{enhanced_edit}")

            with st.spinner("Editing with DeepFloyd IF / SDXL..."):
                edited_versions = []
                for i in range(num_edits):
                    out_bytes = run_if_edit(enhanced_edit, base_bytes)
                    edited_versions.append(out_bytes)

                for idx, out_bytes in enumerate(edited_versions):
                    fname = f"edit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}.png"
                    st.session_state.edited_images.append({"filename": fname, "content": out_bytes})
                    st.image(Image.open(BytesIO(out_bytes)), caption=fname, use_container_width=True)
                    st.download_button("⬇️ Download", data=out_bytes, file_name=fname, mime="image/png", key=f"dl_edit_{idx}")

# ---------------- HISTORY ----------------
st.subheader("📂 History")
if st.session_state.generated_images:
    st.markdown("### Generated")
    for i, img in enumerate(reversed(st.session_state.generated_images[-10:])):
        with st.expander(f"{img['filename']}"):
            st.image(Image.open(BytesIO(img["content"])), caption=img["filename"], use_container_width=True)
if st.session_state.edited_images:
    st.markdown("### Edited")
    for i, img in enumerate(reversed(st.session_state.edited_images[-10:])):
        with st.expander(f"{img['filename']}"):
            st.image(Image.open(BytesIO(img["content"])), caption=img["filename"], use_container_width=True)
