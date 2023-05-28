from pydantic import BaseModel, Field
from typing import Optional, List

class ChatResponse(BaseModel):
    response: str = Field(...)
    context_referenced: Optional[List[str]] = Field(default=None)
    response_code: int = Field(...)
