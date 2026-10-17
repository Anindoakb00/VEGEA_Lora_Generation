import json

with open("flux_lora_v2_master_pipeline.ipynb") as f:
    nb = json.load(f)

# Fix double backslash in regex and remove extra comment in OPTIONAL REFINE cell
for i, line in enumerate(nb["cells"][7]["source"]):
    if "match = re.search(r'_" in line:
        nb["cells"][7]["source"][i] = "                        match = re.search(r'_(\\d+)\\.\\w+$', img_path.name)\n"

# Remove the extra comment line in the last cell
nb["cells"][-1]["source"] = [
    "# OPTIONAL REFINE\n",
]

with open("flux_lora_v2_master_pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
