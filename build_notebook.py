import nbformat as nbf
import uuid

def create_cell_id():
    return uuid.uuid4().hex[:8]

nb = nbf.v4.new_notebook()

# -----------------
# IMPORTS & CONSTANTS
# -----------------
code_setup = """import os
import sys
from pathlib import Path

# Settings
IMAGES_PER_STEP = 8
EVAL_STEPS = [250, 500, 750, 1000, 1250, 1500]

TEST_PROMPTS = [
    "vga_founder, short stubble, neutral expression, looking at the camera, wearing a plain black crew-neck t-shirt, white seamless studio backdrop, soft even studio lighting",
    "vga_founder, short stubble, neutral expression, looking at the camera, wearing a white t-shirt, standing outdoors, natural daylight",
    "vga_founder, short stubble, smiling with teeth, looking at the camera, wearing a light blue linen shirt, outdoors, natural daylight",
    "vga_founder, neutral expression, looking at the camera, wearing a plain black crew-neck t-shirt, white seamless studio backdrop, soft even studio lighting",
    "vga_founder, short stubble, neutral expression, looking at the camera, wearing a black blazer over a plain black crew-neck t-shirt, standing in three-quarter view, light lavender-grey studio backdrop, soft even studio lighting",
    "vga_founder, short stubble, smiling with teeth, looking at the camera, wearing a plain black crew-neck t-shirt, seated at a black table with hands clasped, white seamless studio backdrop, soft even studio lighting",
    "vga_founder, short stubble, neutral expression, looking at the camera, wearing a dark navy v-neck t-shirt, close-up portrait outdoors, weathered white wall, soft natural daylight",
    "vga_founder, neutral expression, looking off to the side, wearing a black waffle-knit long-sleeve sweater, leaning against stacked white sacks, industrial warehouse, dim ambient indoor lighting",
    "vga_founder, short stubble, soft smile, looking away to the left, wearing a plain black crew-neck t-shirt, standing in profile, dark grey studio backdrop, soft side lighting",
    "vga_founder, short stubble, neutral expression, looking at the camera, wearing a light blue linen shirt with a grey knit sweater draped around his neck, restaurant terrace, warm early-evening light"
]

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("Warning: HF_TOKEN environment variable is not set. You may need it to download FLUX models.")
"""
cell_setup = nbf.v4.new_code_cell(code_setup, id=create_cell_id())


# -----------------
# CELL 1: INSTALL
# -----------------
code_install = """# CELL 1: INSTALL
import subprocess
import sys

def install_packages():
    print("Installing packages...")
    packages = ["insightface", "onnxruntime-gpu", "opencv-python-headless", "pandas", "numpy", "pillow", "diffusers", "transformers", "accelerate", "peft", "sentencepiece", "safetensors"]

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during installation.")
        print("If insightface failed to build, you may need to install build dependencies:")
        print("  sudo apt-get update && sudo apt-get install -y build-essential cmake")
        print("  pip install cython")
        print("Then try again.")

install_packages()
"""
cell_install = nbf.v4.new_code_cell(code_install, id=create_cell_id())


# -----------------
# CELL 2: DATASET CHECK AND CLEANUP (safe)
# -----------------
code_dataset = """# CELL 2: DATASET CHECK AND CLEANUP
import shutil
import os
from pathlib import Path

dataset_dir = Path("/workspace/dataset")
if not dataset_dir.exists():
    print(f"Error: Dataset directory {dataset_dir} does not exist.")
else:
    # 1. Count images and txts
    images = [f for f in dataset_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    txts = [f for f in dataset_dir.iterdir() if f.is_file() and f.suffix.lower() == '.txt']

    # Assert counts
    if len(images) != 24 or len(txts) != 24:
        print(f"Error: Expected exactly 24 images and 24 texts. Found {len(images)} images and {len(txts)} texts.")

    errors_found = False

    # 2. Match names
    image_stems = {f.stem for f in images}
    txt_stems = {f.stem for f in txts}

    unpaired_images = image_stems - txt_stems
    unpaired_txts = txt_stems - image_stems

    if unpaired_images:
        print(f"Error: Images without matching txt files: {unpaired_images}")
        errors_found = True
    if unpaired_txts:
        print(f"Error: Txt files without matching image files: {unpaired_txts}")
        errors_found = True

    stray_files = [f for f in dataset_dir.iterdir() if f.is_file() and f.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.txt']]
    if stray_files:
        print(f"Warning: Stray files found: {[f.name for f in stray_files]}")

    # 3. Check captions
    short_stubble_count = 0
    light_stubble_count = 0
    clean_shaven_count = 0

    for txt_file in txts:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content.startswith("vga_founder,"):
                print(f"Error: Caption in {txt_file.name} does not start with 'vga_founder,'. Content: '{content[:30]}...'")
                errors_found = True

            content_lower = content.lower()
            if "short stubble" in content_lower:
                short_stubble_count += 1
            if "light stubble" in content_lower:
                light_stubble_count += 1
            if "clean-shaven" in content_lower:
                clean_shaven_count += 1

    print(f"Caption stats: 'short stubble': {short_stubble_count}, 'light stubble': {light_stubble_count}, 'clean-shaven': {clean_shaven_count}")

    if errors_found:
        raise ValueError("Errors found in dataset. Please fix them before continuing.")
    else:
        print("Dataset check passed successfully!")

    # 4. Clean up latent cache
    print("\\nChecking for latent cache folders to remove...")
    for item in dataset_dir.iterdir():
        if item.is_dir() and "latent" in item.name.lower():
            print(f"Dry-run: Would delete directory {item}")
            # Uncomment the next line to actually delete
            shutil.rmtree(item)
            print(f"Deleted directory {item}")

    # 5. Check disk space
    stat = shutil.disk_usage("/workspace")
    free_gb = stat.free / (1024**3)
    if free_gb < 15:
        print(f"WARNING: Low disk space! Only {free_gb:.2f} GB free on /workspace.")
    else:
        print(f"Disk space OK: {free_gb:.2f} GB free.")

    # 6. Move old run output
    old_out = Path("/workspace/output/founder_v2_lora")
    new_out = Path("/workspace/output/founder_v2_lora_old")
    if old_out.exists():
        if not new_out.exists():
            print(f"Moving {old_out} to {new_out} to prevent resuming...")
            shutil.move(str(old_out), str(new_out))
        else:
            print(f"Note: {old_out} exists, but {new_out} also exists. Skipping move.")
"""
cell_dataset = nbf.v4.new_code_cell(code_dataset, id=create_cell_id())


# -----------------
# CELL 3: CONFIG
# -----------------
code_config = """# CELL 3: CONFIG
import yaml
import copy
from pathlib import Path
import difflib

# Find ai-toolkit run.py
workspace = Path("/workspace")
ai_toolkit_path = None
for p in workspace.rglob("run.py"):
    # Assuming run.py in ai-toolkit root
    if p.parent.name == "ai-toolkit" or (p.parent / "toolkit").exists():
        ai_toolkit_path = p.parent
        break

print(f"Found ai-toolkit at: {ai_toolkit_path}")

old_config_path = Path("/workspace/output/founder_v2_lora_old/config.yaml") # It was moved in cell 2
if not old_config_path.exists():
    # Try finding it globally if the specific path fails
    for p in workspace.rglob("config.yaml"):
        if "founder_v2_lora" in str(p):
            old_config_path = p
            break

if not old_config_path.exists():
    print(f"Error: Could not find existing config.yaml.")
else:
    print(f"Loading base config from {old_config_path}")
    with open(old_config_path, 'r') as f:
        old_config = yaml.safe_load(f)

    new_config = copy.deepcopy(old_config)

    # We assume standard structure based on Ostris ai-toolkit config
    config_key = list(new_config['config'].keys())[0] if 'config' in new_config and new_config['config'] else 'process'

    process = new_config.get('config', {}).get(config_key, {})
    if not process:
        print("Warning: Could not parse standard config structure.")

    process['name'] = 'founder_v3_lora'

    # Update model settings
    if 'model' in process:
        process['model']['train_text_encoder'] = False
        process['model']['resolution'] = [512, 768, 1024]

    # Update network settings (Keep rank 16, etc. untouched as requested, just making sure the block exists)

    # Update dataset settings
    if 'datasets' in process:
        dataset_list = process['datasets']
        if isinstance(dataset_list, list) and len(dataset_list) > 0:
            dataset_list[0]['folder_path'] = '/workspace/dataset'

    # Update train settings
    if 'train' in process:
        process['train']['steps'] = 1500
        process['train']['save_every'] = 100
        process['train']['lr'] = '1e-4'
        # max_step_saves_to_keep might be in save/train

    process['trigger_word'] = 'vga_founder'

    # Save options
    if 'save' in process:
         process['save']['max_step_saves_to_keep'] = 15
    elif 'train' in process:
         process['train']['max_step_saves_to_keep'] = 15

    # Update sample settings
    if 'sample' in process:
        process['sample']['sample_every'] = 250
        process['sample']['seed'] = 42
        process['sample']['walk_seed'] = False
        process['sample']['guidance_scale'] = 3.5
        process['sample']['sample_steps'] = 28
        process['sample']['width'] = 1024
        process['sample']['height'] = 1024

        # Take first IMAGES_PER_STEP prompts
        process['sample']['prompts'] = TEST_PROMPTS[:IMAGES_PER_STEP]

    # Write back
    new_config_path = Path("/workspace/config_v3.yaml")
    with open(new_config_path, 'w') as f:
        yaml.dump(new_config, f, sort_keys=False)

    print(f"Saved new config to {new_config_path}")

    # Print diff
    with open(old_config_path, 'r') as f1, open(new_config_path, 'r') as f2:
        diff = difflib.unified_diff(f1.readlines(), f2.readlines(), fromfile='old', tofile='new')
        print("".join(diff))
"""
cell_config = nbf.v4.new_code_cell(code_config, id=create_cell_id())


# -----------------
# CELL 4: LAUNCH TRAINING AND SAMPLE ORGANIZER
# -----------------
code_train = """# CELL 4: LAUNCH TRAINING AND SAMPLE ORGANIZER
import os
import subprocess
import shlex
from pathlib import Path

# Write sample organizer script
organizer_script = '''import time
import shutil
import re
from pathlib import Path

source_dir = Path("/workspace/output/founder_v3_lora/samples")
eval_dir = Path("/workspace/eval/training_samples")

print("Starting sample organizer...")
while True:
    try:
        if source_dir.exists():
            current_time = time.time()
            for img_path in source_dir.glob("*.jpg"): # or png
                # Skip files modified in the last 10 seconds (might still be writing)
                if current_time - img_path.stat().st_mtime < 10:
                    continue

                # Parse 9-digit step
                match = re.search(r'_(\d{9})', img_path.name)
                if match:
                    step_str = match.group(1)
                    step = int(step_str)

                    target_dir = eval_dir / f"step_{step}"
                    target_dir.mkdir(parents=True, exist_ok=True)

                    target_file = target_dir / img_path.name
                    if not target_file.exists():
                        print(f"Copying {img_path.name} to {target_dir}")
                        shutil.copy2(img_path, target_file)

        time.sleep(30)
    except Exception as e:
        print(f"Error in organizer: {e}")
        time.sleep(30)
'''

with open("/workspace/organize_samples.py", "w") as f:
    f.write(organizer_script)

print("Starting sample organizer in background...")
organizer_cmd = "n\\o\\h\\u\\p python3 /workspace/organize_samples.py > /workspace/organize_samples.log 2>&1 &"
os.system(organizer_cmd)

print("Starting training in background...")
# Assuming ai-toolkit path is found in previous cell. If not, fallback to /workspace/ai-toolkit
ai_path = ai_toolkit_path if 'ai_toolkit_path' in locals() and ai_toolkit_path else "/workspace/ai-toolkit"

# Launch ai-toolkit
train_cmd = f"cd {ai_path} && n\\o\\h\\u\\p python3 run.py /workspace/config_v3.yaml > /workspace/train_v3.log 2>&1 &"
os.system(train_cmd)

print("Training launched. Use the monitor cell below to track progress.")
"""
cell_train = nbf.v4.new_code_cell(code_train, id=create_cell_id())


# -----------------
# CELL 4b: MONITOR
# -----------------
code_monitor = """# CELL 4b: MONITOR TRAINING
import os
import subprocess
from pathlib import Path

log_file = Path("/workspace/train_v3.log")
if log_file.exists():
    print("--- Last 10 lines of training log ---")
    try:
        tail = subprocess.check_output(['tail', '-n', '10', str(log_file)]).decode()
        print(tail)
    except Exception as e:
        print(f"Error reading log: {e}")
else:
    print(f"Log file {log_file} does not exist yet.")

eval_dir = Path("/workspace/eval/training_samples")
if eval_dir.exists():
    print("\\n--- Sample Files Collected ---")
    for step_dir in sorted(eval_dir.iterdir(), key=lambda x: int(x.name.split('_')[-1]) if '_' in x.name else 0):
        if step_dir.is_dir():
            count = len(list(step_dir.glob("*.*")))
            print(f"{step_dir.name}: {count} files")
"""
cell_monitor = nbf.v4.new_code_cell(code_monitor, id=create_cell_id())


# -----------------
# CELL 5: POST-TRAINING INFERENCE
# -----------------
code_infer = """# CELL 5: POST-TRAINING INFERENCE
import os
import gc
import torch
from pathlib import Path
from PIL import Image
try:
    from diffusers import FluxPipeline
except ImportError:
    print("Warning: Diffusers not installed or too old. Please ensure it is installed.")

# Free GPU memory
print("Freeing GPU memory...")
gc.collect()
torch.cuda.empty_cache()

output_model_dir = Path("/workspace/output/founder_v3_lora/founder_v3_lora")
eval_dir = Path("/workspace/eval")

def find_checkpoint(step):
    step_str = f"{step:09d}"

    # Try exact match
    matches = list(output_model_dir.glob(f"*_{step_str}.safetensors"))
    if matches:
         return matches[0]

    # Fallback for 1500 (final un-numbered)
    if step == 1500:
        matches = list(output_model_dir.glob("*.safetensors"))
        un_numbered = [m for m in matches if not any(c.isdigit() for c in m.stem.split('_')[-1])]
        if un_numbered:
            return un_numbered[0]

    return None

def build_contact_sheet(images, cols=4, title_height=30):
    if not images:
        return None
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

if not output_model_dir.exists():
    print(f"Error: Output dir {output_model_dir} not found. Did training complete?")
else:
    print("Loading FLUX pipeline...")
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            token=HF_TOKEN
        ).to("cuda")

        contact_sheets_dir = eval_dir / "contact_sheets"
        contact_sheets_dir.mkdir(parents=True, exist_ok=True)

        for step in EVAL_STEPS:
            ckpt_path = find_checkpoint(step)
            if not ckpt_path:
                print(f"Warning: Checkpoint for step {step} not found. Skipping.")
                continue

            print(f"\\nProcessing step {step} from {ckpt_path.name}")

            ckpt_out_dir = eval_dir / f"checkpoint_{step}"
            ckpt_out_dir.mkdir(parents=True, exist_ok=True)

            pipe.load_lora_weights(str(ckpt_path), adapter_name="default")

            step_images = []

            for i, prompt in enumerate(TEST_PROMPTS[:IMAGES_PER_STEP]):
                out_file = ckpt_out_dir / f"prompt_{i+1}_seed_1.png"
                if out_file.exists():
                    print(f"  Skipping {out_file.name} (already exists)")
                    try:
                        step_images.append(Image.open(out_file))
                    except:
                        pass
                    continue

                print(f"  Generating prompt {i+1}...")
                generator = torch.Generator(device="cpu").manual_seed(1)

                # Try-except for generation to be safe
                try:
                    img = pipe(
                        prompt=prompt,
                        height=1024,
                        width=1024,
                        num_inference_steps=28,
                        guidance_scale=3.5,
                        joint_attention_kwargs={"scale": 1.0},
                        generator=generator
                    ).images[0]
                    img.save(out_file)
                    step_images.append(img)
                except Exception as e:
                    print(f"  Error generating: {e}")

            # Build contact sheet for checkpoint
            if step_images:
                sheet = build_contact_sheet(step_images)
                if sheet:
                    sheet_path = contact_sheets_dir / f"step_{step}_ckpt.jpg"
                    # Resize to avoid huge files
                    sheet.thumbnail((2048, 2048))
                    sheet.save(sheet_path)
                    print(f"  Saved contact sheet {sheet_path.name}")

            pipe.unload_lora_weights("default")

            # Free memory between checkpoints
            gc.collect()
            torch.cuda.empty_cache()

        # Build contact sheets for training samples
        print("\\nBuilding contact sheets for training samples...")
        training_samples_dir = eval_dir / "training_samples"
        if training_samples_dir.exists():
            for step_dir in training_samples_dir.iterdir():
                if step_dir.is_dir():
                    imgs = []
                    for p in sorted(step_dir.glob("*.*")):
                        try:
                            imgs.append(Image.open(p))
                        except:
                            pass
                    if imgs:
                        sheet = build_contact_sheet(imgs)
                        if sheet:
                            sheet_path = contact_sheets_dir / f"{step_dir.name}_train.jpg"
                            sheet.thumbnail((2048, 2048))
                            sheet.save(sheet_path)
                            print(f"  Saved contact sheet {sheet_path.name}")

    except Exception as e:
        print(f"Pipeline error: {e}")
"""
cell_infer = nbf.v4.new_code_cell(code_infer, id=create_cell_id())


# -----------------
# CELL 6: ARCFACE EVALUATION
# -----------------
code_eval = """# CELL 6: ARCFACE EVALUATION
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
try:
    from insightface.app import FaceAnalysis
except ImportError:
    print("Insightface not installed. Evaluation cannot proceed.")

# Initialize FaceAnalysis
print("Initializing InsightFace...")
try:
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
except Exception as e:
    print(f"Error initializing InsightFace: {e}")
    app = None

def get_face_embedding(img_path, app):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    faces = app.get(img)
    if not faces:
        return None
    # Largest face
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
    emb = faces[0].embedding
    # L2 normalize
    emb = emb / np.linalg.norm(emb)
    return emb

if app:
    dataset_dir = Path("/workspace/dataset")
    eval_dir = Path("/workspace/eval")

    print("Calculating identity centroid...")
    real_embeddings = []
    skipped_reals = []

    # 1. Identity centroid
    for img_path in dataset_dir.glob("*.*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            emb = get_face_embedding(img_path, app)
            if emb is not None:
                real_embeddings.append(emb)
            else:
                skipped_reals.append(img_path.name)

    if skipped_reals:
         print(f"Skipped dataset images (no face detected): {skipped_reals}")

    if not real_embeddings:
        print("Error: No faces found in dataset!")
    else:
        # Average and L2 normalize
        centroid = np.mean(real_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        # Reference table
        print("\\nReference Table (Similarity to Centroid):")
        ref_sims = []
        for i, emb in enumerate(real_embeddings):
            sim = np.dot(emb, centroid)
            ref_sims.append(sim)
            # print(f"  Real {i}: {sim:.4f}") # Too spammy

        print(f"  Mean Reference Similarity (Realistic Ceiling): {np.mean(ref_sims):.4f}")
        print(f"  Min Reference Similarity: {np.min(ref_sims):.4f}")

        # 2. Score eval images
        print("\\nScoring evaluation images...")
        results = []

        # Helper to process a directory
        def process_dir(dir_path, source_type, step_val):
            if not dir_path.exists():
                return

            sims = []
            valid = 0
            total = 0
            prompt_sims = {}

            for img_path in sorted(dir_path.glob("*.*")):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    total += 1
                    emb = get_face_embedding(img_path, app)

                    # Extract prompt info if possible (e.g., prompt_1)
                    prompt_idx = "unknown"
                    if "prompt_" in img_path.name:
                        parts = img_path.name.split("_")
                        try:
                            idx = parts.index("prompt") + 1
                            prompt_idx = f"prompt_{parts[idx]}"
                        except:
                            pass

                    if prompt_idx not in prompt_sims:
                        prompt_sims[prompt_idx] = []

                    if emb is not None:
                        sim = np.dot(emb, centroid)
                        sims.append(sim)
                        prompt_sims[prompt_idx].append(sim)
                        valid += 1
                    else:
                        # Count no-face as 0 similarity or just mark as failure?
                        # "Count no-face images as failures, do not drop them silently."
                        sims.append(0.0) # Penalty for no face
                        prompt_sims[prompt_idx].append(0.0)

            if total > 0:
                mean_sim = np.mean(sims)
                min_sim = np.min(sims)
                std_sim = np.std(sims)
                det_rate = valid / total

                row = {
                    'step': step_val,
                    'source': source_type,
                    'mean': mean_sim,
                    'min': min_sim,
                    'std': std_sim,
                    'det_rate': det_rate,
                }

                for p_idx, p_sims in prompt_sims.items():
                    if p_sims:
                        row[f'mean_{p_idx}'] = np.mean(p_sims)

                results.append(row)

        # Process checkpoints
        for step_dir in eval_dir.glob("checkpoint_*"):
            try:
                step = int(step_dir.name.split("_")[1])
                process_dir(step_dir, "checkpoint", step)
            except:
                pass

        # Process training samples
        training_samples_dir = eval_dir / "training_samples"
        if training_samples_dir.exists():
            for step_dir in training_samples_dir.glob("step_*"):
                try:
                    step = int(step_dir.name.split("_")[1])
                    process_dir(step_dir, "training_sample", step)
                except:
                    pass

        # 3. Output DataFrame
        if results:
            df = pd.DataFrame(results)

            # Compute a combined ranking score.
            # "Rank by mean, demoting any step with detection rate below 90%."

            # Combine both sources per step
            # For simplicity, we can groupby step and average the means, or just rank them directly.
            # The prompt says: "combine both sources for each step into one ranking"

            combined = df.groupby('step').agg({
                'mean': 'mean',
                'det_rate': 'mean'
            }).reset_index()

            # Demote if det_rate < 0.90 by drastically reducing score
            combined['rank_score'] = combined.apply(
                lambda row: row['mean'] if row['det_rate'] >= 0.90 else row['mean'] - 1.0,
                axis=1
            )

            combined = combined.sort_values(by='rank_score', ascending=False).reset_index(drop=True)

            res_file = eval_dir / "results.csv"
            df.to_csv(res_file, index=False)
            print(f"\\nResults saved to {res_file}")

            print("\\nTop 3 Steps:")
            for i, row in combined.head(3).iterrows():
                step = int(row['step'])
                mean_val = row['mean']
                det = row['det_rate']
                print(f"  {i+1}. Step {step}: Mean Similarity = {mean_val:.4f}, Det Rate = {det:.1%}")
                print(f"     -> /workspace/eval/checkpoint_{step}/")
                print(f"     -> /workspace/eval/training_samples/step_{step}/")
"""
cell_eval = nbf.v4.new_code_cell(code_eval, id=create_cell_id())


# -----------------
# FINAL CELL: LAYOUT
# -----------------
code_final = """# FINAL CELL
import os
from pathlib import Path

eval_dir = Path("/workspace/eval")
print("\\n--- Output Folder Layout ---")

total_images = 0

def print_tree(dir_path, prefix=""):
    global total_images
    if not dir_path.exists():
         return

    contents = sorted(dir_path.iterdir())
    dirs = [d for d in contents if d.is_dir()]
    files = [f for f in contents if f.is_file()]

    print(f"{prefix}{dir_path.name}/")

    # Print file count
    img_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    total_images += len(img_files)
    if img_files:
        print(f"{prefix}  └─ {len(img_files)} images")
    elif files:
        print(f"{prefix}  └─ {len(files)} files")

    for d in dirs:
        print_tree(d, prefix + "  ")

if eval_dir.exists():
    print_tree(eval_dir)
else:
    print("Eval directory not created yet.")

print(f"\\nTotal Images Generated: {total_images}")

# Read top from CSV if exists
res_file = eval_dir / "results.csv"
top_step = None
if res_file.exists():
    import pandas as pd
    try:
        df = pd.read_csv(res_file)
        # simplistic recalculation to find top step
        combined = df.groupby('step').agg({'mean': 'mean', 'det_rate': 'mean'}).reset_index()
        combined['rank_score'] = combined.apply(
            lambda row: row['mean'] if row['det_rate'] >= 0.90 else row['mean'] - 1.0, axis=1
        )
        top_step = int(combined.sort_values('rank_score', ascending=False).iloc[0]['step'])
    except:
        pass

if top_step:
    print(f"\\nRECOMMENDED CHECKPOINT: Step {top_step}")
else:
    print("\\nRECOMMENDED CHECKPOINT: Check results.csv")

print("\\nREMINDER: Please review the images from the recommended checkpoint by eye before delivery. ArcFace measures identity similarity, not overall image quality, lighting, or prompt adherence.")
"""
cell_final = nbf.v4.new_code_cell(code_final, id=create_cell_id())


# Assemble notebook
nb.cells = [cell_setup, cell_install, cell_dataset, cell_config, cell_train, cell_monitor, cell_infer, cell_eval, cell_final]

with open('flux_lora_v2_master_pipeline.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook flux_lora_v2_master_pipeline.ipynb created.")
