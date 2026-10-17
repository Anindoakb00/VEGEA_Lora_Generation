import json

with open("flux_lora_v2_master_pipeline.ipynb") as f:
    nb = json.load(f)

failures = 0
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        try:
            compile("".join(cell["source"]), f"cell_{i}", "exec")
        except Exception as e:
            print(f"Compilation error in cell {i+1}: {e}")
            failures += 1

print(f"Compilation check finished with {failures} failures.")
