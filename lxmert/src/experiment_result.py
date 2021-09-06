import glob
import json
import os

result = {}
count = 1
for name in glob.glob("models/*/*.json"):
    print(name)
    if 'test_predict' in name:
        continue
    
    os.system(f'cp {name} result/')
    file_ = open(name)
    content = json.load(file_)

    result[name.replace('models/', '')] = content

with open("result/lxmert_experiment.json", "w", encoding="utf8") as json_file :
    json.dump(result, json_file, ensure_ascii=False, indent=4)

for name in glob.glob("models/*/*.log"):
    os.system(f'cp {name} logs/')
