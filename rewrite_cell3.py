import json

with open('flux_lora_v2_master_pipeline.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell.get('id') == '3eef5f97':
        src = ''.join(cell['source'])
        start_marker = "    # We assume standard structure based on Ostris ai-toolkit config"
        end_marker = "    # Write back"
        if start_marker in src and end_marker in src:
            before = src[:src.find(start_marker)]
            after = src[src.find(end_marker):]

            replacement = """    cfg = new_config['config']
    cfg['name'] = 'founder_v3_lora'
    process = cfg['process'][0]
    process['trigger_word'] = 'vga_founder'
    process['save']['save_every'] = 100
    process['save']['max_step_saves_to_keep'] = 15
    process['train']['steps'] = 1500
    process['train']['lr'] = 0.0001
    process['train']['train_text_encoder'] = False
    process['datasets'][0]['folder_path'] = '/workspace/dataset'
    process['datasets'][0]['resolution'] = [512, 768, 1024]
    process['sample']['sample_every'] = 250
    process['sample']['prompts'] = TEST_PROMPTS[:IMAGES_PER_STEP]

"""
            cell['source'] = (before + replacement + after).splitlines(True)

with open('flux_lora_v2_master_pipeline.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
