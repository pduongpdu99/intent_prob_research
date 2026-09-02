import nltk
from engine.Util import get_vietnamese_stopwords, get_non_sw
from typing import cast, List

try:
    # Try to look up the resource to see if it exists
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    # Download it only if it is missing
    nltk.download('punkt_tab')

def extract_tokens(
    document: str,
    pass_stop_word=False,
    ngram=4
) -> List[str]:
    results = []

    for sent in document.split("."):
        sent = sent.strip()

        sw = get_vietnamese_stopwords()
        nsw = cast(dict, get_non_sw(kv_structure=True))

        tokens = nltk.word_tokenize(sent.lower())
        _size = len(tokens)

        if not pass_stop_word:
            results.append(tokens)

        result = []
        index = 0

        while index < _size:
            val = tokens[index]

            # Skip punctuation
            if val in ",.!@,?><[];':\\":
                index += 1
                continue

            # --------------------------------------------------
            # 1. Nếu token có trong non-stopword dictionary
            #    -> thử tìm n-gram
            # --------------------------------------------------
            if val in nsw:
                temps = []
                current_phrase = ""

                for i in range(ngram):
                    next_idx = index + i

                    if next_idx >= _size:
                        break

                    current_phrase += (
                        tokens[next_idx]
                        if i == 0
                        else " " + tokens[next_idx]
                    )

                    temps.append((current_phrase, i + 1))

                # Ưu tiên n-gram dài nhất
                temps.reverse()

                matched = False

                for phrase, jump in temps:
                    if phrase in nsw[val]:
                        result.append(phrase)
                        index += jump
                        matched = True
                        break

                if matched:
                    continue

                # --------------------------------------------------
                # Không match được n-gram
                #
                # -> kiểm tra token đơn có phải stopword không
                # -> nếu không phải thì giữ lại
                # -> luôn index += 1
                # --------------------------------------------------
                if val not in sw:
                    result.append(val)

                index += 1

            # --------------------------------------------------
            # 2. Token không có trong nsw
            #    -> chỉ cần kiểm tra stopword
            # --------------------------------------------------
            else:
                if val not in sw:
                    result.append(val)

                index += 1

        results.append(result)

    return results


def detect_entities(tokens):
    from engine.constants import ENTITY_MAP
    result=[]
    emlist = list(ENTITY_MAP.keys())
    for tok in tokens:
        if tok in emlist:
            result.append((tok, ENTITY_MAP[tok]))
            continue
    return result

def detect_triggers(tokens: list[str]) -> list[tuple[str, str]]:
    from engine.constants import TRIGGER_VERBS
    tvlist = list(TRIGGER_VERBS.keys())
    triggers = []
    for tok in tokens:
        if tok in tvlist:
            triggers.append((tok, TRIGGER_VERBS[tok]))
    return triggers

def build_relations(entities, triggers):
    from engine.constants import RELATION_RULES
    rrlist = list(RELATION_RULES.keys())
    relations = []
    
    # đơn giản: lấy từng cặp entity theo thứ tự xuất hiện
    for i, (e1_text, e1_type) in enumerate(entities):
        for j, (e2_text, e2_type) in enumerate(entities):
            if i >= j:
                continue
            
            key = (e1_type, e2_type)
            if key not in rrlist:
                continue
            
            rule = RELATION_RULES[key]
            rel_type = rule["default"]
            
            # nếu có trigger nằm giữa 2 entity thì override
            for trig_text, trig_label in triggers:
                if trig_text in rule["triggers"]:
                    rel_type = rule["triggers"][trig_text]
                    break
            
            relations.append({
                "source": e1_text,
                "source_type": e1_type,
                "relation": rel_type,
                "target": e2_text,
                "target_type": e2_type
            })
    
    return relations
