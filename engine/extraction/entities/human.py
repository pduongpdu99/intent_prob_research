from pydantic import BaseModel
from typing import Optional
class HumanEntity(BaseModel):
    name: Optional[str]
    age: Optional[str]
    address: Optional[str]
    occupation: Optional[str]