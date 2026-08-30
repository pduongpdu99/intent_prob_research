from xml.sax.xmlreader import AttributesImpl
from engine.constants import Tag
from typing import List
import xml.sax

def create_parser(handler: XMLHandler, only_parser = True):
    parser = xml.sax.make_parser()
    parser.setContentHandler(handler)
    if only_parser: return parser
    return parser, handler._categories


class XMLHandler(xml.sax.ContentHandler):
    _stack:List[str] = []
    _categories: List[dict] = []
    _current_template: List[dict] = []
    _pattern_links = {}
    _content = ""

    def startElement(self, name: str, attrs: AttributesImpl) -> None:
        self.add_content()
        if name == Tag.CATEGORY.value:
            self._stack.append(name)
        elif name == Tag.PATTERN.value:
            self._stack.append(name)
        elif name == Tag.TEMPLATE.value:
            self._stack.append(name)
        elif name == Tag.SRAI.value:
            self.add_content()
            node = {
                "tag": "srai",
                "refer": ""
            }
            self._current_template.append(node)
        elif name == Tag.STAR.value:
            node = {
                "tag": "star",
                "index": int(attrs.get("index", "1"))
            }
            self._current_template.append(node)
        elif name == Tag.SET.value:
            if not attrs.get("name"):
                raise KeyError("XML key name can not null")
            
            node = {"tag": "set","key": attrs.get("name"),"value": attrs.get("value", "Unknown")}
            self._current_template.append(node)
        elif name == Tag.GET.value:
            node = {"tag": "get","key": attrs.get("name", "Unknown")}
            self._current_template.append(node)
            pass

    def endElement(self, name: str) -> None:
        if name == Tag.SRAI.value:
            self._pattern_links[self.current_pattern] = self._content

        if name == Tag.CATEGORY.value:
            self._categories.append({
                "pattern": self.current_pattern,
                "template": self._current_template
            })
            self._stack.pop()
            self.flush()
        elif name == Tag.PATTERN.value:
            self.current_pattern = self._content
            self.clean_str()
            self._stack.pop()
        elif name == Tag.TEMPLATE.value:
            self.add_content()
            self._stack.pop()
        elif name == Tag.SRAI.value:
            _size = len(self._current_template)
            t = self._current_template[_size-1]
            if _size>0:
                t['refer']=self._content
                self._current_template[_size-1] = t
            self.clean_str()
        elif name == Tag.SET.value:
            val = self._content
            _size = len(self._current_template)
            t = self._current_template[_size-1]
            if _size>0 and val.strip() != "":
                t['value']=val
                self._current_template[_size-1] = t
            self.clean_str()
        elif name == Tag.GET.value:
            pass
    
    def characters(self, content:str) -> None:
        self._content += content.strip()

    def add_content(self):
        self._content = self._content.strip()
        if self._content != "":
            self._current_template.append({
                "tag": "text",
                "value": self._content
            })
            self.clean_str()

    def clean_str(self):
        self._content = ""
    
    def flush(self):
        self.memory = {}
        self._stack = []
        self.current_pattern = ""
        self._current_template = []
        self.clean_str()

        