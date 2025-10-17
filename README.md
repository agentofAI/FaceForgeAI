---
title: FaceForgeAI ZeroGPU
emoji: 🐨
colorFrom: pink
colorTo: pink
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: FaceForgeAI_ZeroGPU
---

# 🎨 FaceForge AI – ZeroGPU Gradio Edition  
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Open%20in-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/VcRlAgent/FaceForgeAI_ZeroGPU)

**Author:** Vijay S. Chaudhari
**Runtime:** Hugging Face Spaces (ZeroGPU) 🚀  
 
---

## 🧠 Overview  
**FaceForge AI** transforms your uploaded photo into professional-quality images powered by open-source generative AI:  
- 🪄 **Background Remover** – clean gradient background for profile photos  
- 🛂 **Passport Photo** – compliant 600×600 white-background image  (Requires picture with frontal face)
- 🎭 **Stylized AI Avatar** – realistic yet personalized stylization  

This edition is optimized for **ZeroGPU Spaces** — efficient, on-demand GPU execution using CPU offload and attention slicing for lightweight inference.

---

## ⚙️ Key Features  
✅ **Three Modes**
- **Background Remover** – removes background and enhances clarity  
- **Passport** – white background, standard size  
- **Avatar** – realistic stylization via Stable Diffusion Img2Img  

✅ **User Control for Avatar Generation**
- Preset prompt styles + custom text input  
- Adjustable `strength` & `guidance_scale` sliders  

---

## 🧩 Tech Stack  

| Component | Purpose |
|------------|----------|
| **Python 3.10+** | Core runtime |
| **Gradio 4.x** | Web UI framework |
| **Stable Diffusion v1.5** | Img2Img stylization |
| **GFPGAN 1.3.8** | Facial restoration |
| **Real-ESRGAN 0.3.0** | Super-resolution |
| **Rembg 2.x** | Background removal |
| **Pillow / OpenCV / Torch** | Image processing + GPU acceleration |


---

## 🧰 Installation  

### 1️⃣ Clone Repository
```bash
git clone https://github.com/agentofAI/FaceForgeAI.git
cd FaceForgeAI_ZeroGPU
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Locally
```bash
python app.py
```
Open the local Gradio URL (typically http://127.0.0.1:7860) in your browser.

---

## 🎨 Avatar Prompt Presets
Style Label	            Prompt
🎬 Cinematic Portrait	highly detailed, digital portrait, professional lighting, cinematic style, artistic AI avatar
🎨 Stylized Realism	    stylized yet realistic portrait, balanced lighting, subtle gradient background, sharp 
                        focus on face
🏢 Studio Professional	studio portrait, even lighting, neutral background, realistic skin, confident pose
🤵 Natural Headshot	    realistic professional headshot, soft studio lighting, neutral background, crisp details, 
                        natural skin tone

---

## 🧾 Model Credits
| Model                     | Source / License                                                                                              |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Stable Diffusion v1.5** | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) (CompVis / Runway ML) |
| **GFPGAN v1.3**           | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) (MIT License)                                       |
| **Real-ESRGAN x2Plus**    | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD License)                                   |
| **Rembg**                 | [danielgatis/rembg](https://github.com/danielgatis/rembg)                                                     |
| **Gradio**                | [gradio-app/gradio](https://github.com/gradio-app/gradio)                                                     |

---

## 🧠 Project Structure
faceforge-ai/
│
├── app.py                # Main application
├── requirements.txt      # Dependencies
├── assets/               # Optional screenshots and samples
└── README.md             # Documentation

---

## 📜 License

This project is for educational and demonstration purposes.
Each model used retains its original open-source license.

---
## 👨‍💻 Author

Vijay S. Chaudhari
