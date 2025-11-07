import json
from comet import load_from_checkpoint

model = load_from_checkpoint("../XCOMET-XL/checkpoints/model.ckpt")


src = "orginal_yue"
mt = "chinese"
ref = "original"

json_data = json.load(open("sentence_translations_google.json", "r", encoding="utf-8"))

data = []

for item in json_data:
    data.append({
        "src": item[src],
        "mt": item[mt],
        "ref": item[ref]
    })

model_output = model.predict(data, batch_size=8, gpus=0)

print(model_output.scores)
print(model_output.system_scores)
print(model_output.metadata)

json.dump(model_output.scores, open("xcomet_xl_google_yue_chinese.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)