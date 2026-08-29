from .handler import create_parser, XMLHandler
from xml.sax.expatreader import ExpatParser
from typing import List
import re
import random
from engine.intent import Util
# call parser 

class Brain:
    pattern = []
    REFLECTIONS = {
        "tôi": "bạn",
        "của tôi": "của bạn",
        "mình": "bạn",
        "bạn": "tôi",
        "của bạn": "của tôi",
    }

    def __init__(self):
        self.__handler = XMLHandler()
        self.__parser = create_parser(self.__handler)
        self.__memory = {}
        self.__parsed_data = None
        self.__pattern_links = self.__handler._pattern_links

    def learn(self, data_path:str):
        self.__exception()
        self.__parser.parse(data_path)
        self.__expose_parsed_data()
        self.__process_parsed_data()

    def respond(self, prompt: str):
        for pattern, responses in self.pattern:
            group = re.findall(pattern, Util.remove_accent(prompt).upper())
            if group:
                gr = [self.__reflect(g) for g in group]
                response = responses if type(responses) is str else random.choice(responses)
                return response

                # return response.format(*gr) if gr else response


    # private methodologies
    def __expose_parsed_data(self):
        self.__parsed_data = self.__handler._categories
    
    def __exception(self, must_be_reader=True):
        if not self.__parser:
            raise ValueError("Parser can not null")

        if must_be_reader and type(self.__parser) is not ExpatParser:
            raise ValueError("{0} must be reader".format("Parser unit"))

    def __process_parsed_data(self):
        self.__exception()
        for item in self.__parsed_data:
            pattern_msg = item['pattern']
            template_list = item['template']

            self.pattern.append(
                (pattern_msg, (self.__process_template_item(pattern_msg, template_list)))
            )

    def __process_template_item(self, pattern:str, template: List[dict]):
        result = []
        for item in template:
            tag =item["tag"]
            if tag == "text":
                result.append(item["value"])
            elif tag == "star":
                result.append("{" + str(item['index']-1) + "}")
            elif tag == "set":
                self.__memory[item['key']] = item['value']
            elif tag == "get":
                result.append(self.__memory[item['key']])
            
            if tag == "srai":
                _p = pattern
                _next = self.__pattern_links[_p]
                while True:
                    _p = _next
                    if _p not in self.__pattern_links:
                        _next = None
                        break
                    _next = self.__pattern_links[_p]
                result.append(_p)
            else:
                if tag == "text":
                    self.__pattern_links[pattern] = item['value']
        return " ".join(result)

    def __reflect(self, text: str):
        words = text.lower().split(" ")
        for i, w in enumerate(words):
            if w in self.REFLECTIONS:
                words[i] = self.REFLECTIONS[w]
        return " ".join(words)

    