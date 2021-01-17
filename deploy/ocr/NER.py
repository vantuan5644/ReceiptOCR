import spacy
import os
from config import PROJECT_ROOT

MODEL_DIR = os.path.join(PROJECT_ROOT, 'pretrained_models/NER')

nlp = spacy.load(MODEL_DIR)

def get_NER_results(text: str):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    # print("Entities", entities)
    tokens = [(t.text, t.ent_type_, t.ent_iob) for t in doc]
    # print("Tokens", tokens)
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities.keys():
            entities[ent.label_] = ent.text
        else:
            entities[ent.label_] += ' ' + ent.text
    print(entities)
    return entities

if __name__ == '__main__':
    text = '04902430418287 NX DOWNY doahoa ngotngao VAT10%   2    129,900.00    259,800.00 \\nGia goc:   159,000.00 '

    print(get_NER_results(text))