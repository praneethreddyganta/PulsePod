import numpy as np
import os

# =============================================================================
# --- CONFIGURATION: INPUT PATHS ---
# =============================================================================
# 1. Path to your MIT-BIH Test Data (from Step 2.5/3)
MIT_PATH = r"V:\Projects\MP\Ver3.0\edge_deployement\processed_data_balanced"

# 2. Path to your PTB-XL Test Data (from the cross_validation_project)
PTB_PATH = r"V:\Projects\MP\Ver3.0\cross_validation_project\ptbxl_processed_local"

# 3. Where to save the new Unified Set
OUTPUT_DIR = r"V:\Projects\MP\Ver3.0\cross_validation_project\data\unified_test_set"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("--- Creating Unified Test Set Locally ---")

# =============================================================================
# --- STEP 1: LOAD DATA ---
# =============================================================================
try:
    print(f"Loading MIT-BIH data from: {MIT_PATH}")
    X_mit = np.load(os.path.join(MIT_PATH, "X_test.npy"))
    y_mit = np.load(os.path.join(MIT_PATH, "y_test.npy"))
    print(f"   -> MIT-BIH Samples: {len(y_mit)}")

    print(f"Loading PTB-XL data from: {PTB_PATH}")
    X_ptb = np.load(r"V:\Projects\MP\Ver3.0\cross_validation_project\ptbxl_processed_local\X_test.npy")
    y_ptb = np.load(r"V:\Projects\MP\Ver3.0\cross_validation_project\ptbxl_processed_local\y_test.npy")
    print(f"   -> PTB-XL Samples:  {len(y_ptb)}")

except FileNotFoundError as e:
    print(f"\n🚨 ERROR: Could not find file. Check your paths.\n{e}")
    exit()

# =============================================================================
# --- STEP 2: COMBINE (CONCATENATE) ---
# =============================================================================
print("\nCombining datasets...")

# np.vstack stacks arrays vertically (adds rows)
X_unified = np.vstack((X_mit, X_ptb))

# np.hstack stacks 1D arrays horizontally (combines labels)
y_unified = np.hstack((y_mit, y_ptb))

# =============================================================================
# --- STEP 3: SAVE ---
# =============================================================================
print(f"Saving unified data to: {OUTPUT_DIR}")

np.save(os.path.join(OUTPUT_DIR, "X_test_unified.npy"), X_unified)
np.save(os.path.join(OUTPUT_DIR, "y_test_unified.npy"), y_unified)

print("\n✅ Success! Unified Test Set Created.")
print(f"   - Final X Shape: {X_unified.shape}")
print(f"   - Final y Shape: {y_unified.shape}")

# Verify Class Balance
unique, counts = np.unique(y_unified, return_counts=True)
print(f"   - Class Distribution: {dict(zip(unique, counts))} (0=Normal, 1=Abnormal)")