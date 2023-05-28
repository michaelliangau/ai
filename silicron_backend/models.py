from pydantic import BaseModel, Field
from typing import Dict, Any


class ChatInput(BaseModel):
    prompt: str
    config: Dict[str, Any]

class User(BaseModel):
    name: str
    email: str