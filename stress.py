from engine.intent.objects import XML
from engine.intent.Util import join


parser,categories = XML.create_parser(
    XML.XMLHandler()
)

parser.parse(join(
    "engine",
    "intent",
    "data",
    # "tu_van_phan_mem.aiml"
    "text.xml"
))

print(categories)