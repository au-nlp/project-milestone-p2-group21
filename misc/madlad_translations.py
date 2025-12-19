from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import json

model_name = "jbochi/madlad400-3b-mt"
model = T5ForConditionalGeneration.from_pretrained(model_name, device_map=None)
tokenizer = T5Tokenizer.from_pretrained(model_name)

ds = pd.read_json(path_or_buf="smoldata/smol_en_yue_sentences/test.json", lines=True)
samples = ds.to_records()
records = []

# this takes about 20 min on CPU
for i, ex in enumerate(samples):
    original = ex['trg'] # type: ignore
    text = f"<2en> {original}" 
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    outputs = model.generate(input_ids=input_ids, max_new_tokens=128)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    record = {
        'original': ex["src"],
        'original_yue': original,
        'cantonese': translation
    }
    
    records.append(record)
    
with open(r"sentence_translations_madlad400-3b.json", 'w+') as f:
    json.dump(records, f, indent=4)