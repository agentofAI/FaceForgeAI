import os
import replicate
from PIL import Image
import io
import base64
import tempfile
from typing import Optional


class ReplicateHandler:
    def __init__(self, model: str, default_settings: dict):
        self.model = model
        self.default_settings = default_settings
        
        # Verify API token
        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            raise ValueError("REPLICATE_API_TOKEN not found in environment variables")
    
    def _image_to_base64_url(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL for Replicate"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def _save_temp_image(self, image: Image.Image) -> str:
        """Save image to temp file and return path"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(temp_file.name, format="PNG")
        return temp_file.name
    
    def generate(
        self,
        input_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        custom_settings: Optional[dict] = None
    ) -> Image.Image:
        """
        Generate image using Replicate InstantID
        
        Args:
            input_image: PIL Image
            prompt: Positive prompt
            negative_prompt: Negative prompt
            custom_settings: Override default settings (cfg, steps, etc.)
        
        Returns:
            Generated PIL Image
        """
        # Merge settings
        settings = {**self.default_settings}
        if custom_settings:
            settings.update(custom_settings)
        
        # Save temp image and get file object
        temp_path = self._save_temp_image(input_image)
        
        try:
            # Prepare input
            input_params = {
                "image": open(temp_path, "rb"),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                **settings
            }
            
            # Run prediction (streaming)
            output = replicate.run(self.model, input=input_params)
            
            # Get final image from iterator
            result_url = None
            for item in output:
                result_url = item  # Last item is the final image URL
            
            if not result_url:
                raise ValueError("No output received from Replicate")
            
            # Download and convert to PIL
            import requests
            response = requests.get(result_url)
            result_image = Image.open(io.BytesIO(response.content))
            
            return result_image
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)