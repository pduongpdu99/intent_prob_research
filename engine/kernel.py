from .Util import (
    ENTITIES_PATH,
    TRIGGERS_PATH,
    RELATION_PATH,
    DATA_COLLECTION_XLSX_PATH,
    SOFTWARE_ROLE_PATH,
    SOFTWARE_DOMAIN_PATH,
    read_json
)
import pandas as pd
from typing import List, cast
from engine.extraction import ExtractionEngine

class Kernel:
    entities_data: List[dict] = []
    triggers_data: List[dict] = []
    relation_data: List[dict] = []
    raci_matrix = pd.read_excel(DATA_COLLECTION_XLSX_PATH, sheet_name="raci")
    raci_matrix_domain = cast(dict, read_json(SOFTWARE_DOMAIN_PATH)).keys()
    raci_matrix_role = cast(dict, read_json(SOFTWARE_ROLE_PATH)).keys()

    # tools
    extraction_tool = ExtractionEngine()

    def __init__(self):
        self.entities_data = read_json(ENTITIES_PATH)
        self.entities_data = read_json(TRIGGERS_PATH)
        self.entities_data = read_json(RELATION_PATH)
        self.__raci_matrix_norm()
        self.raci_matrix_domain = self.raci_matrix.index

    def __raci_matrix_norm(self):
        keyname = "Unnamed: 0"
        self.raci_matrix.set_index(self.raci_matrix[keyname],inplace=True)
    
    def create_relation(self):
        self.extraction_tool.export_from_template()

    def create_base(self):
        self.extraction_tool.export_knowledge_base_json()