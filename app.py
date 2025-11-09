# ==========================================
# FaceForge AI – ZeroGPU Gradio Version
# Author: Vijay S. Chaudhari | 2025
# ==========================================

import gradio as gr
import spaces
import torch
import cv2
import numpy as np
from pathlib import Path

from PIL import Image, ImageEnhance, ImageOps
from rembg import remove
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionXLPipeline
import io
import os, sys, subprocess, warnings, logging

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# --- Ensure InstantID is available ---
if not Path("instantid").exists():
    print("🔄 Cloning InstantID repository...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/InstantID/InstantID.git", "instantid"],check=True)

repo_root = Path("instantid").resolve()

# 🧭 Search for a pipeline file that matches *instantid*.py under the repo
candidates = list(repo_root.rglob("pipeline*instantid*.py"))
if not candidates:
    # Fallback common names across commits
    fallback_names = [
        "pipelines/pipeline_instantid.py",
        "pipelines/pipeline_stable_diffusion_instantid.py",
        "pipelines/pipeline_stable_diffusion_xl_instantid.py",
    ]
    for name in fallback_names:
        p = repo_root / name
        if p.exists():
            candidates = [p]
            break

if not candidates:
    raise FileNotFoundError(
        "Could not locate an InstantID pipeline file under ./instantid. "
        "Repo layout may have changed. Please check the repo structure."
    )

pipeline_file = candidates[0]
print(f"✅ Using InstantID pipeline file: {pipeline_file.relative_to(repo_root)}")

# 🪄 Import the pipeline module by file path (no package needed)
spec = importlib.util.spec_from_file_location("instantid_pipeline", str(pipeline_file))
instantid_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(instantid_mod)  # type: ignore

# 🔎 Pick a pipeline class that looks like an InstantID Pipeline
InstantIDPipeline = None
for attr in dir(instantid_mod):
    if "InstantID" in attr and "Pipeline" in attr:
        InstantIDPipeline = getattr(instantid_mod, attr)
        break

if InstantIDPipeline is None:
    # Helpful diagnostics
    print("Available names in module:", [a for a in dir(instantid_mod) if "Pipeline" in a])
    raise ImportError(
        "Could not find an InstantID pipeline class. "
        "Looked for a class name containing both 'InstantID' and 'Pipeline'."
    )

print(f"✅ Imported pipeline class: {InstantIDPipeline.__name__}")

'''
if os.path.exists("InstantID") and not os.path.exists("instantid"):
    os.rename("InstantID", "instantid")    

instantid_path = os.path.abspath("instantid")
sys.path.append(instantid_path)
sys.path.append(os.path.join(instantid_path, "pipelines"))

#sys.path.append(os.path.abspath("instantid"))
#sys.path.insert(0, os.path.join(os.getcwd(), 'InstantID'))
try:
    from pipelines.pipeline_instantid import InstantIDPipeline
    print("✅ InstantIDPipeline imported successfully.")
except Exception as e:
    print("⚠️ Failed to import InstantIDPipeline:", e)
    InstantIDPipeline = None  # graceful fallback
'''

import torchvision
print("Printing Torch and TorchVision versions:")
print(torch.__version__)
print(torchvision.__version__)

# GPU libraries
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# ------------------------------------------
# Model Loading (Outside GPU decorator)
# ------------------------------------------

def load_models():
    """Load models once at startup"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # RealESRGAN upsampler
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device
    )
    
    # GFPGAN enhancer
    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler,
        device=device
    )
    
    # Stable Diffusion Img2Img pipeline (public model)
    sd_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)

    # Optimize for ZeroGPU memory
    sd_pipe.enable_attention_slicing()
    sd_pipe.enable_model_cpu_offload()

    return face_enhancer, sd_pipe

# Load models globally
face_enhancer, sd_pipe = load_models()

# ------------------------------------------
# GPU-Accelerated Functions
# ------------------------------------------

@spaces.GPU
def enhance_face(img: Image.Image) -> Image.Image:
    """Enhance face using GFPGAN (GPU)"""
    img_cv = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
    
    with torch.no_grad():
        _, _, restored_img = face_enhancer.enhance(
            img_cv,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=0.5
        )
    
    restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(restored_img)

# ------------------------------------------
# Image Processing Functions
# ------------------------------------------

def enhance_image(img: Image.Image) -> Image.Image:
    """Basic enhancement"""
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.1)
    return img

@spaces.GPU
def create_headshot(img: Image.Image) -> Image.Image:
    """Professional headshot with gradient background"""
    # Enhance face
    img_enhanced = enhance_face(img)
    
    # Remove background
    img_no_bg = remove(img_enhanced)
    
    # Gradient background
    bg = Image.new("RGB", img_no_bg.size, (200, 210, 230))
    if img_no_bg.mode == 'RGBA':
        bg.paste(img_no_bg, mask=img_no_bg.split()[3])
    
    return enhance_image(bg)

@spaces.GPU
def create_passport(img: Image.Image) -> Image.Image:
    """Passport photo with white background"""
    # Enhance face
    img_enhanced = enhance_face(img)
    
    # Remove background
    img_no_bg = remove(img_enhanced)
    
    # White background (600x600)
    bg = Image.new("RGB", (600, 600), (255, 255, 255))
    img_no_bg.thumbnail((550, 550), Image.Resampling.LANCZOS)
    offset = ((600 - img_no_bg.width) // 2, (600 - img_no_bg.height) // 2)
    
    if img_no_bg.mode == 'RGBA':
        bg.paste(img_no_bg, offset, mask=img_no_bg.split()[3])
    
    return bg

@spaces.GPU
def create_avatar(img: Image.Image, prompt: str, strength: float, guidance_scale: float) -> Image.Image:
    """
    Create a stylized AI avatar while preserving facial identity using InstantID.
    Retains core facial features, skin tone, and expressions of the input photo.
    """

    # Stylize with SD prompt. We are selecting these from UI now.
    #prompt = "highly detailed, digital portrait, professional lighting, cinematic style, artistic AI avatar"
    #prompt = "stylized yet realistic portrait, balanced lighting, subtle gradient background, sharp focus on face"
    #prompt = "studio portrait, even lighting, neutral background, realistic skin, confident pose"
    #prompt = "realistic professional headshot, soft studio lighting, neutral background, crisp details, natural skin tone"    

    # --- Convert input ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = img.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)

    # --- Step 1: Load InstantID + SDXL pipeline ---
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16
    ).to(device)

    instantid = InstantIDPipeline.from_pretrained("InstantID/InstantID", torch_dtype=torch.float16,)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    #pipe.load_ip_adapter(instantid)

    # --- Step 2: Optimize for ZeroGPU memory ---
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()

    # --- Step 3: Prepare conditioning (face embedding) ---
    np_img = np.array(img)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    face_emb = instantid.extract_face_embedding(bgr_img)  # key step: ID embedding guidance

    # --- Step 4: Stylized generation ---
    gen = pipe.generate_with_identity(
        image=img,
        face_embedding=face_emb,
        prompt=(
            prompt
            + ", portrait of the same person, consistent identity, detailed lighting, "
              "highly realistic skin texture, cinematic color tones"
        ),
        strength=float(strength),
        guidance_scale=float(guidance_scale),
        num_inference_steps=30
    )

    avatar = gen.images[0]

    # --- Step 5 (Optional): Post-process with GFPGAN for crispness ---
    try:
        from gfpgan import GFPGANer
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=device
        )
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler,
            device=device
        )

        img_cv = cv2.cvtColor(np.array(avatar), cv2.COLOR_RGB2BGR)
        _, _, restored_img = face_enhancer.enhance(
            img_cv, has_aligned=False, only_center_face=False,
            paste_back=True, weight=0.4
        )
        avatar = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"[WARN] GFPGAN post-process skipped: {e}")

    return avatar


@spaces.GPU
def process_all(img: Image.Image):
    """Process all three types at once"""
    headshot = create_headshot(img)
    passport = create_passport(img)
    avatar = create_avatar(img)
    return headshot, passport, avatar

# ------------------------------------------
# Gradio Interface
# ------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(), title="FaceForge AI") as demo:
    gr.Markdown(
        """
        # 🎨 FaceForge AI
        ### GPU-Accelerated Professional Headshot & Avatar Generator  
        Upload your photo and choose or customize how your AI avatar is generated.
        """
    )

    # --- Define a mapping: Short Label -> Full Prompt Text ---
    PROMPT_MAP = {
        "🎬 Cinematic Portrait": "highly detailed, digital portrait, professional lighting, cinematic style, artistic AI avatar",
        "🎨 Stylized Realism": "stylized yet realistic portrait, balanced lighting, subtle gradient background, sharp focus on face",
        "🏢 Studio Professional": "studio portrait, even lighting, neutral background, realistic skin, confident pose",
        "🤵 Natural Headshot": "realistic professional headshot, soft studio lighting, neutral background, crisp details, natural skin tone"
    }

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📷 Upload Your Photo")

            gr.Markdown("### ⚙️ Avatar Generation Settings")

            # Dropdown shows short labels only
            preset_prompt = gr.Dropdown(
                label="🎨 Choose Avatar Style Preset",
                choices=list(PROMPT_MAP.keys()),
                value="🤵 Natural Headshot"
            )

            # Optional custom prompt box for flexibility
            custom_prompt = gr.Textbox(
                label="✏️ Custom Prompt (optional)",
                placeholder="Enter your own prompt or leave blank to use preset...",
                lines=2
            )

            strength_slider = gr.Slider(
                label="🎛️ Style Strength (0.0 = keep original, 1.0 = full restyle)",
                minimum=0.1,
                maximum=1.0,
                value=0.45,
                step=0.05
            )

            guidance_slider = gr.Slider(
                label="🎯 Prompt Guidance Scale (higher = more prompt influence)",
                minimum=1.0,
                maximum=10.0,
                value=5.5,
                step=0.5
            )

        with gr.Column(scale=1):
            gr.Markdown("### Results")

            # --- Independent Outputs & Buttons ---
            output_headshot = gr.Image(label="💼 Professional Headshot", type="pil")
            btn_headshot = gr.Button("📸 Generate Headshot", variant="secondary")

            output_passport = gr.Image(label="🛂 Passport Photo", type="pil")
            btn_passport = gr.Button("🪪 Generate Passport", variant="secondary")

            output_avatar = gr.Image(label="🎭 AI Avatar", type="pil")
            btn_avatar = gr.Button("✨ Generate Avatar", variant="primary")

    # --- Functions for individual generations ---
    def run_headshot(img):
        return create_headshot(img)

    def run_passport(img):
        return create_passport(img)

    def run_avatar(img, preset_label, custom, strength, guidance):
        final_prompt = custom.strip() if custom and custom.strip() != "" else PROMPT_MAP[preset_label]
        return create_avatar(img, final_prompt, strength, guidance)

    # --- Button actions ---
    btn_headshot.click(fn=run_headshot, inputs=[input_image], outputs=[output_headshot])
    btn_passport.click(fn=run_passport, inputs=[input_image], outputs=[output_passport])
    btn_avatar.click(
        fn=run_avatar,
        inputs=[input_image, preset_prompt, custom_prompt, strength_slider, guidance_slider],
        outputs=[output_avatar]
    )
    
    gr.Markdown(
        """
        ---
        ### Features
        - 💼 **Professional Headshots**: Perfect for LinkedIn and business profiles
        - 🛂 **Passport Photos**: Standard 600x600px with white background
        - 🎭 **AI Avatars**: Stylized versions for social media
        - ⚡ **GPU-Accelerated**: Fast processing with GFPGAN enhancement
        
        © 2025 Vijay S. Chaudhari | Powered by ZeroGPU 🚀
        """
    )

# Launch
if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch()