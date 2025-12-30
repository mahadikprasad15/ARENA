#!/usr/bin/env python3
"""
Script to update the Harmfulness_and_Refusal_Probes.ipynb notebook with fixes:
1. Fix create_expanded_harmless_csv() to save to correct location
2. Add Google Drive integration for persistent storage
"""

import json
import shutil
from pathlib import Path

# Backup original
shutil.copy(
    '/home/user/ARENA/Harmfulness_and_Refusal_Probes.ipynb',
    '/home/user/ARENA/Harmfulness_and_Refusal_Probes.ipynb.backup'
)

# Read notebook
with open('/home/user/ARENA/Harmfulness_and_Refusal_Probes.ipynb', 'r') as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells")

# ==============================================================================
# FIX 1: Update create_expanded_harmless_csv() to save to correct location
# ==============================================================================

# Find cell 2 with download functions
download_cell = nb['cells'][2]
source = ''.join(download_cell['source'])

# Fix 1: Change catqa path to xstest path in create_expanded_harmless_csv
if 'def create_expanded_harmless_csv()' in source:
    print("\n✓ Found create_expanded_harmless_csv function")

    # Replace the incorrect path
    old_path_line = '    os.makedirs("data/catqa", exist_ok=True)'
    new_path_line = '    # *** FIXED: Save to xstest directory so load_xstest() can find it ***\\n    os.makedirs("data/xstest", exist_ok=True)'

    source = source.replace(old_path_line, new_path_line)

    old_filepath = '    filepath = "data/catqa/questions.csv"'
    new_filepath = '    filepath = "data/xstest/xstest_v2_prompts.csv"  # *** FIXED: Use xstest path ***'

    source = source.replace(old_filepath, new_filepath)

    print("  ✓ Fixed file path to save to data/xstest/xstest_v2_prompts.csv")

    # Update cell source
    download_cell['source'] = [source]
else:
    print("⚠ Warning: create_expanded_harmless_csv function not found")

# ==============================================================================
# FIX 2: Add Google Drive integration cell
# ==============================================================================

drive_cell_source = """# ==============================================================================
# Google Drive Integration for Persistent Storage
# ==============================================================================

def mount_google_drive():
    \"\"\"Mount Google Drive for persistent storage in Colab.

    Returns:
        str: Path to cache directory (Drive if available, local otherwise)
    \"\"\"
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted at /content/drive")

        # Create cache directory in Drive
        drive_cache = '/content/drive/MyDrive/harmfulness_probe_cache'
        os.makedirs(drive_cache, exist_ok=True)
        print(f"✓ Cache directory: {drive_cache}")
        print(f"  Files saved here will persist across sessions!")

        return drive_cache
    except ImportError:
        print("⚠ Not running in Google Colab - using local storage")
        os.makedirs('cache', exist_ok=True)
        return 'cache'
    except Exception as e:
        print(f"✗ Failed to mount Drive: {e}")
        print("  Using local storage (will be deleted on disconnect)")
        os.makedirs('cache', exist_ok=True)
        return 'cache'


def save_activations_to_drive(compressed_data, filename, drive_cache_dir=None):
    \"\"\"Save compressed activations to Google Drive for persistence.

    Args:
        compressed_data: List of MinimalActs to save
        filename: Name of pickle file (e.g., 'harmful_inst_acts.pkl')
        drive_cache_dir: Drive cache directory (if None, will mount Drive)

    Returns:
        str: Path where file was saved
    \"\"\"
    if drive_cache_dir is None:
        drive_cache_dir = mount_google_drive()

    filepath = os.path.join(drive_cache_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(compressed_data, f)

    file_size_mb = os.path.getsize(filepath) / 1024 / 1024
    print(f"✓ Saved {len(compressed_data)} activations to Drive")
    print(f"  Location: {filepath}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  This file will persist even after GPU disconnect!")

    # Also save to local cache as backup
    local_path = os.path.join('cache', filename)
    os.makedirs('cache', exist_ok=True)
    with open(local_path, 'wb') as f:
        pickle.dump(compressed_data, f)
    print(f"  Backup: {local_path} (temporary)")

    return filepath


def load_activations_from_drive(filename, drive_cache_dir=None):
    \"\"\"Load compressed activations from Google Drive.

    Args:
        filename: Name of pickle file to load
        drive_cache_dir: Drive cache directory (if None, will mount Drive)

    Returns:
        List of MinimalActs
    \"\"\"
    if drive_cache_dir is None:
        drive_cache_dir = mount_google_drive()

    filepath = os.path.join(drive_cache_dir, filename)

    # Try Drive first
    if os.path.exists(filepath):
        print(f"Loading from Drive: {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded {len(data)} activations from Drive")
        return data

    # Fallback to local cache
    local_path = os.path.join('cache', filename)
    if os.path.exists(local_path):
        print(f"⚠ Drive file not found, loading from local cache: {local_path}")
        print(f"  Warning: Local cache is temporary!")
        with open(local_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ Loaded {len(data)} activations")
        return data

    raise FileNotFoundError(
        f"File not found in Drive ({filepath}) or local cache ({local_path})\\n"
        f"Available files in Drive: {os.listdir(drive_cache_dir) if os.path.exists(drive_cache_dir) else '[]'}"
    )


print("✓ Google Drive integration functions loaded")
print()
print("=" * 80)
print("IMPORTANT: How to Use Persistent Storage")
print("=" * 80)
print()
print("# Step 1: Mount Google Drive (run this FIRST in your session)")
print("drive_cache = mount_google_drive()")
print()
print("# Step 2: Save activations to Drive (they will persist!)")
print("save_activations_to_drive(compressed_data, 'my_acts.pkl', drive_cache)")
print()
print("# Step 3: Later (even after disconnect), load from Drive")
print("compressed_data = load_activations_from_drive('my_acts.pkl', drive_cache)")
print()
print("=" * 80)
"""

# Find where to insert the Drive cell (after download cell, before data classes)
insert_index = 3  # After cell 0 (imports), cell 1 (more setup), cell 2 (downloads)

drive_cell = {
    "cell_type": "code",
    "source": [drive_cell_source],
    "metadata": {"id": "drive_integration"},
    "execution_count": None,
    "outputs": []
}

nb['cells'].insert(insert_index, drive_cell)
print(f"\n✓ Inserted Google Drive integration cell at position {insert_index}")

# ==============================================================================
# Save updated notebook
# ==============================================================================

with open('/home/user/ARENA/Harmfulness_and_Refusal_Probes.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print(f"\n✓ Notebook updated successfully!")
print(f"  Total cells: {len(nb['cells'])}")
print(f"  Backup saved: Harmfulness_and_Refusal_Probes.ipynb.backup")
print()
print("FIXES APPLIED:")
print("  1. ✓ Fixed create_expanded_harmless_csv() to save to data/xstest/")
print("  2. ✓ Added Google Drive mount/save/load functions")
print()
print("Now your notebook will:")
print("  • Download balanced 250+250 datasets correctly")
print("  • Save activations to Google Drive for persistence")
print("  • Load activations even after GPU disconnect")
