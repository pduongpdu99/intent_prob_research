from engine.intent.objects import handler
from engine.intent.Util import join


parser,categories = handler.create_parser(
    handler.XMLHandler()
)

parser.parse(join(
    "engine",
    "intent",
    "data",
    # "tu_van_phan_mem.aiml"
    "text.xml"
))

print(categories)