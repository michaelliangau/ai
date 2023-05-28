from pydantic import BaseModel, Field
from typing import Dict, Any


class ChatInput(BaseModel):
    api_key: str
    prompt: str
    config: Dict[str, Any]