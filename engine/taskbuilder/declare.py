from pydantic import BaseModel
from typing import List
import pandas as pd

class TaskBuilderPayload(BaseModel):
    entities:List[dict]
    triggers:List[dict]
    relation:List[dict]
    matrix:pd.DataFrame
    