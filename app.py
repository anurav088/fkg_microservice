from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
from quantulum3 import parser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model"

nlp = spacy.load(MODEL_PATH)  

app = FastAPI(title="Ingredient Parser Microservice")

class TextRequest(BaseModel):
    text: str

@app.post("/annotate_ingredients")
def annotate_ingredients_endpoint(request: TextRequest):
    try:
        doc = nlp(request.text)
        if not doc.ents:
            return {"entities": []}
        return {
            "entities": [
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                } for ent in doc.ents
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract_modifiers_and_ingredient")
def extract_modifiers_and_ingredient_endpoint(request: TextRequest):
    try:
        ents = annotate_ingredients_endpoint(request)["entities"] or []

        qty_ent = next((e for e in ents if e['label'] == 'QUANTITY'), None)
        unit_ent = next((e for e in ents if e['label'] == 'UNIT'), None)
        state_ent = next((e for e in ents if e['label'] == 'STATE'), None)
        ing_ents = [e for e in ents if e['label'] == 'ING']

        modifiers = []
        if qty_ent and unit_ent and qty_ent['start'] < unit_ent['start']:
            modifiers.append(request.text[qty_ent['start']:unit_ent['end']])
        elif qty_ent:
            modifiers.append(qty_ent['text'])
        if state_ent:
            modifiers.append(state_ent['text'])

        if ing_ents:
            ing = " ".join(e['text'] for e in sorted(ing_ents, key=lambda x: x['start']))
        else:
            spans_to_remove = []
            if qty_ent and unit_ent:
                spans_to_remove.append((qty_ent['start'], unit_ent['end']))
            elif qty_ent:
                spans_to_remove.append((qty_ent['start'], qty_ent['end']))
            if state_ent:
                spans_to_remove.append((state_ent['start'], state_ent['end']))
            
            chars = list(request.text)
            for st, en in sorted(spans_to_remove, reverse=True):
                for i in range(st, en):
                    chars[i] = ""
            ing = "".join(chars).strip(" ,;:").strip()
            if ing.lower().startswith("of "):
                ing = ing[3:]

        return {"modifiers": modifiers, "ingredient": ing}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse_quantity")
def parse_quantity_endpoint(request: TextRequest):
    try:
        quantities = parser.parse(request.text)  
        if quantities:
            return {"quantity": round(float(quantities[0].value), 2)}
        return {"quantity": None}
    except Exception:
        return {"quantity": None}


@app.post("/extract_unit_info")
def extract_unit_info_endpoint(request: TextRequest):
    try:
        mod, ingredient = extract_modifiers_and_ingredient_endpoint(request).values()
        ents = annotate_ingredients_endpoint(request)["entities"]
        unit, quantity = None, None
        if ents:
            for e in ents:
                if e['label'] == 'UNIT':
                    unit = e['text']
                    break
        quantity = parse_quantity_endpoint(request)["quantity"]
        return {"ingredient": ingredient, "quantity": quantity, "unit": unit}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
