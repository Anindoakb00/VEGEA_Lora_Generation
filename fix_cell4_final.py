import json

with open('flux_lora_v2_master_pipeline.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == '2a70f82e':
        src = ''.join(cell['source'])

        # We know exactly how it looks in python string escaping:
        # organizer_cmd = "n\o\h\u\p
        # and so on
        # Let's do string replacement taking into account the exact raw string representation
        lines = []
        for line in cell['source']:
            # replace the weird nohup
            if "n\\o\\h\\u\\p" in line:
                line = line.replace("n\\o\\h\\u\\p", "nohup")
            if "n\\\\o\\\\h\\\\u\\\\p" in line:
                line = line.replace("n\\\\o\\\\h\\\\u\\\\p", "nohup")
            # fix the \d syntax warning
            if "r'_(\\d{9})'" in line:
                line = line.replace("r'_(\\d{9})'", "r'_(\\\\d{9})'")
            lines.append(line)
        cell['source'] = lines

with open('flux_lora_v2_master_pipeline.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
