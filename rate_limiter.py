import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Tuple
import uuid
import hashlib


class RateLimiter:
    def __init__(self, session_file: str, daily_limit: int, dev_daily_limit: int):
        self.session_file = Path(session_file)
        self.daily_limit = daily_limit
        self.dev_daily_limit = dev_daily_limit
        self.is_dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"
        
        # Create session file if doesn't exist
        if not self.session_file.exists():
            self._save_data({})
    
    def _load_data(self) -> dict:
        """Load rate limit data from file"""
        try:
            with open(self.session_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    
    def _save_data(self, data: dict):
        """Save rate limit data to file"""
        with open(self.session_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _get_device_id(self, request: dict) -> str:
        """Generate consistent device ID from request headers"""
        # Use IP + User-Agent for fingerprinting
        ip = request.get("client", {}).get("host", "unknown")
        user_agent = request.get("headers", {}).get("user-agent", "unknown")
        
        # Hash to create stable ID
        fingerprint = f"{ip}:{user_agent}"
        return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    
    def _get_next_reset(self) -> datetime:
        """Get next midnight UTC"""
        now = datetime.now(timezone.utc)
        tomorrow = now + timedelta(days=1)
        return tomorrow.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _cleanup_expired(self, data: dict) -> dict:
        """Remove expired entries"""
        now = datetime.now(timezone.utc)
        cleaned = {}
        
        for device_id, info in data.items():
            reset_time = datetime.fromisoformat(info["reset_time"])
            if reset_time > now:
                cleaned[device_id] = info
        
        return cleaned
    
    def check_limit(self, request: dict) -> Tuple[bool, int, datetime]:
        """
        Check if device has exceeded rate limit
        
        Returns:
            (allowed: bool, remaining: int, reset_time: datetime)
        """
        device_id = self._get_device_id(request)
        data = self._load_data()
        data = self._cleanup_expired(data)
        
        limit = self.dev_daily_limit if self.is_dev_mode else self.daily_limit
        now = datetime.now(timezone.utc)
        
        if device_id not in data:
            # New device
            reset_time = self._get_next_reset()
            data[device_id] = {
                "count": 0,
                "reset_time": reset_time.isoformat()
            }
            self._save_data(data)
        
        device_info = data[device_id]
        reset_time = datetime.fromisoformat(device_info["reset_time"])
        
        # Check if reset needed
        if now >= reset_time:
            device_info["count"] = 0
            device_info["reset_time"] = self._get_next_reset().isoformat()
            self._save_data(data)
        
        current_count = device_info["count"]
        remaining = max(0, limit - current_count)
        allowed = current_count < limit
        
        return allowed, remaining, reset_time
    
    def increment(self, request: dict):
        """Increment usage count for device"""
        device_id = self._get_device_id(request)
        data = self._load_data()
        
        if device_id in data:
            data[device_id]["count"] += 1
            self._save_data(data)
    
    def get_limit_message(self, remaining: int, reset_time: datetime) -> str:
        """Generate user-friendly limit message"""
        mode = "DEV" if self.is_dev_mode else "Standard"
        limit = self.dev_daily_limit if self.is_dev_mode else self.daily_limit
        
        if remaining > 0:
            return f"✅ {remaining}/{limit} generations remaining today ({mode} mode)"
        else:
            now = datetime.now(timezone.utc)
            hours_left = int((reset_time - now).total_seconds() / 3600)
            minutes_left = int(((reset_time - now).total_seconds() % 3600) / 60)
            
            return f"❌ Daily limit reached ({limit}/{limit}). Resets in {hours_left}h {minutes_left}m (midnight UTC)"