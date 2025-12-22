import gradio as gr
import yaml
import torch
from PIL import Image
from rembg import remove
from rate_limiter import RateLimiter
from replicate_handler import ReplicateHandler


# ------------------------------------------
# Load Configuration
# ------------------------------------------

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize modules
rate_limiter = RateLimiter(
    session_file=config["rate_limit"]["session_file"],
    daily_limit=config["rate_limit"]["default_daily_limit"],
    dev_daily_limit=config["rate_limit"]["dev_daily_limit"]
)

replicate_handler = ReplicateHandler(
    model=config["replicate"]["model"],
    default_settings=config["replicate"]["default_settings"]
)

# ------------------------------------------
# Core Functions
# ------------------------------------------

def remove_background(img: Image.Image) -> Image.Image:
    """Remove background from image"""
    if img is None:
        raise gr.Error("Please upload an image first")
    return remove(img)


def generate_headshot(
    img: Image.Image,
    headshot_style: str,
    use_advanced: bool,
    cfg: float,
    steps: int,
    denoise: float,
    request: gr.Request
) -> Image.Image:
    """Generate professional headshot with selected style"""
    
    # Check rate limit
    allowed, remaining, reset_time = rate_limiter.check_limit(request)
    if not allowed:
        raise gr.Error(rate_limiter.get_limit_message(remaining, reset_time))
    
    if img is None:
        raise gr.Error("Please upload an image first")
    
    # Get prompt from config
    style_config = config["headshots"][headshot_style]
    prompt = style_config["prompt"]
    negative_prompt = style_config["negative"]
    
    # Prepare settings
    custom_settings = None
    if use_advanced:
        custom_settings = {
            "cfg": cfg,
            "steps": steps,
            "denoise": denoise
        }
    
    # Generate
    try:
        result = replicate_handler.generate(
            input_image=img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            custom_settings=custom_settings
        )
        
        # Increment usage
        rate_limiter.increment(request)
        
        return result
        
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


def generate_scene(
    img: Image.Image,
    scene_style: str,
    use_advanced: bool,
    cfg: float,
    steps: int,
    denoise: float,
    request: gr.Request
) -> Image.Image:
    """Generate scene change with selected style"""
    
    # Check rate limit
    allowed, remaining, reset_time = rate_limiter.check_limit(request)
    if not allowed:
        raise gr.Error(rate_limiter.get_limit_message(remaining, reset_time))
    
    if img is None:
        raise gr.Error("Please upload an image first")
    
    # Get prompt from config
    style_config = config["scenes"][scene_style]
    prompt = style_config["prompt"]
    negative_prompt = style_config["negative"]
    
    # Prepare settings
    custom_settings = None
    if use_advanced:
        custom_settings = {
            "cfg": cfg,
            "steps": steps,
            "denoise": denoise
        }
    
    # Generate
    try:
        result = replicate_handler.generate(
            input_image=img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            custom_settings=custom_settings
        )
        
        # Increment usage
        rate_limiter.increment(request)
        
        return result
        
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


def get_rate_limit_status(request: gr.Request) -> str:
    """Get current rate limit status"""
    allowed, remaining, reset_time = rate_limiter.check_limit(request)
    return rate_limiter.get_limit_message(remaining, reset_time)


# ------------------------------------------
# Gradio Interface
# ------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(), title="FaceForge AI") as demo:
    gr.Markdown(
        """
        # 🎨 FaceForge AI
        ### AI-Powered Professional Photo Generator
        Upload your photo and transform it into professional headshots or place yourself in different scenes.
        """
    )
    
    # Rate limit status
    rate_status = gr.Textbox(
        label="📊 Usage Status",
        interactive=False,
        value="Loading..."
    )
    
    with gr.Row():
        # Left Column - Input
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="📷 Upload Your Photo")
            
            gr.Markdown("### ⚙️ Advanced Settings (Optional)")
            
            use_advanced = gr.Checkbox(
                label="Enable Advanced Settings",
                value=False
            )
            
            with gr.Group(visible=True) as advanced_group:
                cfg_slider = gr.Slider(
                    label="🎛️ CFG Scale",
                    minimum=config["replicate"]["advanced_ranges"]["cfg"]["min"],
                    maximum=config["replicate"]["advanced_ranges"]["cfg"]["max"],
                    value=config["replicate"]["default_settings"]["cfg"],
                    step=config["replicate"]["advanced_ranges"]["cfg"]["step"]
                )
                
                steps_slider = gr.Slider(
                    label="🔄 Steps",
                    minimum=config["replicate"]["advanced_ranges"]["steps"]["min"],
                    maximum=config["replicate"]["advanced_ranges"]["steps"]["max"],
                    value=config["replicate"]["default_settings"]["steps"],
                    step=config["replicate"]["advanced_ranges"]["steps"]["step"]
                )
                
                denoise_slider = gr.Slider(
                    label="✨ Denoise",
                    minimum=config["replicate"]["advanced_ranges"]["denoise"]["min"],
                    maximum=config["replicate"]["advanced_ranges"]["denoise"]["max"],
                    value=config["replicate"]["default_settings"]["denoise"],
                    step=config["replicate"]["advanced_ranges"]["denoise"]["step"]
                )
            
            # Toggle visibility of advanced settings
            use_advanced.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[use_advanced],
                outputs=[advanced_group]
            )
        
        # Right Column - Outputs
        with gr.Column(scale=1):
            
            # Background Remover Section
            gr.Markdown("### 🎭 Background Remover")
            output_bg_removed = gr.Image(label="Background Removed", type="pil")
            btn_bg_remove = gr.Button("🗑️ Remove Background", variant="secondary")
            
            gr.Markdown("---")
            
            # Professional Headshots Section
            gr.Markdown("### 💼 Professional Headshots")
            headshot_style = gr.Dropdown(
                label="Choose Headshot Style",
                choices=list(config["headshots"].keys()),
                value=list(config["headshots"].keys())[0]
            )
            output_headshot = gr.Image(label="Professional Headshot", type="pil")
            btn_headshot = gr.Button("📸 Generate Headshot", variant="primary")
            
            gr.Markdown("---")
            
            # Scene Changer Section
            gr.Markdown("### 🌍 Scene Changer")
            scene_style = gr.Dropdown(
                label="Choose Scene",
                choices=list(config["scenes"].keys()),
                value=list(config["scenes"].keys())[0]
            )
            output_scene = gr.Image(label="Scene Changed", type="pil")
            btn_scene = gr.Button("🎬 Change Scene", variant="primary")
    
    # Button Actions
    btn_bg_remove.click(
        fn=remove_background,
        inputs=[input_image],
        outputs=[output_bg_removed]
    )
    
    btn_headshot.click(
        fn=generate_headshot,
        inputs=[
            input_image,
            headshot_style,
            use_advanced,
            cfg_slider,
            steps_slider,
            denoise_slider
        ],
        outputs=[output_headshot]
    )
    
    btn_scene.click(
        fn=generate_scene,
        inputs=[
            input_image,
            scene_style,
            use_advanced,
            cfg_slider,
            steps_slider,
            denoise_slider
        ],
        outputs=[output_scene]
    )
    
    # Update rate limit status on load and after each generation
    demo.load(fn=get_rate_limit_status, inputs=None, outputs=[rate_status])
    btn_headshot.click(fn=get_rate_limit_status, inputs=None, outputs=[rate_status])
    btn_scene.click(fn=get_rate_limit_status, inputs=None, outputs=[rate_status])
    
    gr.Markdown(
        """
        ---
        ### ✨ Features
        - 🎭 **Background Remover**: Clean background removal for any image
        - 💼 **Professional Headshots**: Studio-quality headshots with multiple styles
        - 🌍 **Scene Changer**: Place yourself in different environments
        - ⚡ **AI-Powered**: InstantID technology for identity preservation
        - 🔒 **Rate Limited**: Fair usage with daily limits        
        
        """
    )

    gr.Markdown(
        """
        ---
        ### ⚠️ Notice
        - This is a **personal experimental project** and has undergone **limited testing**. Please use with caution.
        - Although **InstantID** is designed to preserve identity, outputs may vary based on **prompts, model weights, and other influencing factors**. 
        
        
        © 2025 Vijay S. Chaudhari | Powered by Replicate, InstantID and HuggingFace 🚀
        """
    )

# Launch
if __name__ == "__main__":
    demo.queue(max_size=20)
    demo.launch()