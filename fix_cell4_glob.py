import json

with open('flux_lora_v2_master_pipeline.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == '2a70f82e':
        lines = []
        for line in cell['source']:
            if 'source_dir.glob("*.jpg"): # or png' in line:
                line = line.replace('source_dir.glob("*.jpg"): # or png', 'list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png")):')
            lines.append(line)
        cell['source'] = lines

with open('flux_lora_v2_master_pipeline.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
