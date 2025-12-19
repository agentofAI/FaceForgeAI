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

# 🎨 FaceForge AI
[![Hugging Face Space](https://img.shields.io/badge/🤗%20Open%20in-Hugging%20Face%20Space-yellow)](https://huggingface.co/spaces/VcRlAgent/FaceForgeAI)

**Author:** Vijay S. Chaudhari  
**Runtime:** Gradio + Replicate API 🚀

---

## 🧠 Overview
**FaceForge AI** transforms your uploaded photo into professional-quality images powered by AI:
- 🎭 **Background Remover** – clean background removal for any image
- 💼 **Professional Headshots** – studio-quality portraits in multiple styles
- 🌍 **Scene Changer** – place yourself in different environments with identity preservation

This edition uses **Replicate's InstantID** for high-quality identity-preserving transformations with configurable rate limiting for cost control.

---

## ✨ Key Features

✅ **Three Generation Modes**
- **Background Remover** – removes background using Rembg
- **Professional Headshots** – Studio, Office, Premium Editorial styles
- **Scene Changer** – Coastal Run, Urban Evening, Forest Trail environments

✅ **User Controls**
- Preset prompt styles via dropdown
- Optional advanced settings (CFG, Steps, Denoise)
- Real-time usage tracking

✅ **Rate Limiting**
- 5 generations/day (standard mode)
- 10 generations/day (dev mode)
- Midnight UTC reset
- Device-based tracking

---

## 🧩 Tech Stack

| Component | Purpose |
|-----------|---------|
| **Python 3.10+** | Core runtime |
| **Gradio 4.x** | Web UI framework |
| **Replicate API** | InstantID model hosting |
| **InstantID** | Identity-preserving image generation |
| **Rembg 2.x** | Background removal |
| **PyYAML** | Configuration management |
| **Pillow** | Image processing |

---

## 🧰 Installation

### 1️⃣ Clone Repository
```bash
git clone https://github.com/agentofAI/FaceForgeAI.git
cd FaceForgeAI
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Configure Environment
```bash
cp .env.example .env
# Edit .env and add your REPLICATE_API_TOKEN
```

Get your Replicate API token at: https://replicate.com/account/api-tokens

### 4️⃣ Run Locally
```bash
python app.py
```
Open the local Gradio URL (typically http://127.0.0.1:7860) in your browser.

---

## 🎨 Style Presets

### Professional Headshots

| Style | Description |
|-------|-------------|
| 📸 **Studio Headshot** | Professional editorial with neutral background, magazine quality |
| 🏢 **Office Setting** | Subtle office background, premium business attire |
| ✨ **Premium Editorial** | High-end minimal studio, composed presence |

### Scene Changer

| Scene | Description |
|-------|-------------|
| 🏖️ **Coastal Run** | Ocean and horizon background, warm sunrise lighting |
| 🌃 **Urban Evening** | City lights, soft ambient night lighting |
| 🌲 **Forest Trail** | Trees and dirt path, diffused outdoor light |

---

## 📁 Project Structure

```
faceforge-ai/
│
├── app.py                  # Main Gradio application
├── config.yaml             # Prompts and settings configuration
├── rate_limiter.py         # Rate limiting logic
├── replicate_handler.py    # Replicate API wrapper
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
├── rate_limits.json        # Auto-generated usage tracking
└── README.md               # Documentation
```

---

## ⚙️ Configuration

### Rate Limits (config.yaml)
```yaml
rate_limit:
  default_daily_limit: 5      # Standard mode
  dev_daily_limit: 10         # Dev mode
  reset_timezone: "UTC"       # Reset at midnight UTC
```

### Dev Mode
Enable higher rate limits in `.env`:
```bash
DEV_MODE=true
```

### Add Custom Styles
Edit `config.yaml` to add new headshot or scene styles:

```yaml
headshots:
  "🎯 Your Style":
    prompt: |
      Your custom prompt here...
      multiple lines supported
    negative: "Things to avoid..."

scenes:
  "🎪 Your Scene":
    prompt: "Scene description..."
    negative: "Things to avoid..."
```

### Advanced Settings
Toggle "Enable Advanced Settings" in UI to control:
- **CFG Scale**: Prompt adherence (1.0-10.0)
- **Steps**: Generation quality (20-50)
- **Denoise**: Transformation strength (0.5-1.0)

---

## 📊 Rate Limiting

- **Tracking**: Server-side session file per device
- **Reset**: Midnight UTC daily
- **Scope**: Shared across all generation functions
- **Override**: Dev mode doubles the limit

Device fingerprinting uses IP + User-Agent for consistent tracking.

---

## 🧾 Model Credits

| Model | Source / License |
|-------|------------------|
| **InstantID** | [zsxkib/instant-id-basic](https://replicate.com/zsxkib/instant-id-basic) via Replicate |
| **Rembg** | [danielgatis/rembg](https://github.com/danielgatis/rembg) (MIT License) |
| **Gradio** | [gradio-app/gradio](https://github.com/gradio-app/gradio) (Apache 2.0) |

---

## 🐛 Troubleshooting

**"REPLICATE_API_TOKEN not found"**
- Verify `.env` file exists with valid token
- Check token at https://replicate.com/account/api-tokens

**"Daily limit reached"**
- Wait for midnight UTC reset
- Enable `DEV_MODE=true` for higher limits

**Generation fails**
- Verify image uploaded successfully
- Check Replicate API status
- Ensure internet connectivity

**Rate limits not working**
- Delete `rate_limits.json` to reset tracking
- Verify server-side file permissions

---

## 📜 License

This project is for educational and demonstration purposes.  
Each model used retains its original open-source license.

---

Author: Vijay S. Chaudhari
© 2025 Vijay S. Chaudhari