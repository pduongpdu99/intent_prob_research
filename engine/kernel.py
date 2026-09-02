from .Util import (
    ENTITIES_PATH,
    TRIGGERS_PATH,
    RELATION_PATH,
    DATA_COLLECTION_XLSX_PATH
)

import json
import pandas as pd
from typing import List

class Kernel:
    entities_data: List[dict] = []
    triggers_data: List[dict] = []
    relation_data: List[dict] = []
    raci_matrix = pd.read_excel(DATA_COLLECTION_XLSX_PATH, sheet_name="raci")
    raci_matrix_domain = []
    raci_matrix_role = ["BUSINESS_ANALYST","UI_UX_DESIGNER","FRONTEND_DEV","BACKEND_DEV","MOBILE_DEV","QA_QC","DEVOPS",]

    def __init__(self):
        with open(ENTITIES_PATH, "r", encoding="utf-8") as file:
            self.entities_data = json.loads(file.read())
        with open(TRIGGERS_PATH, "r", encoding="utf-8") as file:
            self.triggers_data = json.loads(file.read())
        with open(RELATION_PATH, "r", encoding="utf-8") as file:
            self.relation_data = json.loads(file.read())

        self.__raci_matrix_norm()
        self.raci_matrix_domain = self.raci_matrix.index

    def __raci_matrix_norm(self):
        keyname = "Unnamed: 0"
        self.raci_matrix.set_index(self.raci_matrix[keyname],inplace=True)
        self.raci_matrix = self.raci_matrix[["BUSINESS_ANALYST","UI_UX_DESIGNER","FRONTEND_DEV","BACKEND_DEV","MOBILE_DEV","QA_QC","DEVOPS",]]