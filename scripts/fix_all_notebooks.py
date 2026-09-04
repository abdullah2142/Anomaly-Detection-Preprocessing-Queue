"""
fix_all_notebooks.py — Apply all code fixes to the 8 training notebooks.

Fixes applied:
1. Replace hardcoded RESCUE_MAP with per-severity get_rescue_map() function
2. Seed the apply_fog function for reproducibility
3. Remove the severe-skip guard from clean notebooks 01 and 03
4. Add baseline emission to clean notebooks 01 and 03
5. Document salt-and-pepper in apply_sensor_noise docstring
"""
import json
import os
import re
import sys

NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'notebooks')

def load_notebook(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f"  ✅ Saved {os.path.basename(path)}")

def get_full_source(nb):
    """Concatenate all code cell sources into one string."""
    parts = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            parts.append(''.join(cell['source']))
    return '\n'.join(parts)

def fix_rescue_map(cell_source_list):
    """
    Replace the static RESCUE_MAP dict with a get_rescue_map(severity, config) function.
    Returns modified source list.
    """
    src = ''.join(cell_source_list)
    
    # Pattern: the old static RESCUE_MAP block
    old_rescue_map = (
        'RESCUE_MAP = {\n'
        '    "low_light": [("CLAHE", apply_clahe), ("Retinex", apply_retinex)],\n'
        '    "gaussian_blur": [("Wiener", lambda img: apply_wiener_deconv(img, sigma=25.0, kernel_size=281))],\n'
        '    "motion_blur": [("Wiener (Motion PSF)", lambda img: apply_motion_wiener_deconv(img, kernel_size=151))],\n'
        '    "sensor_noise": [("NLM Denoise", apply_nlm_denoise)],\n'
        '    "fog_haze": [("Dehaze (Dark Channel)", apply_dark_channel_prior_dehaze)]\n'
        '}'
    )
    
    new_rescue_map = (
        'def get_rescue_map(severity, config):\n'
        '    """Return rescue methods with PSF parameters matched to the current corruption severity.\n'
        '    \n'
        '    This ensures Wiener deconvolution uses the exact PSF that generated the blur,\n'
        '    rather than a hardcoded severe-tier kernel.\n'
        '    """\n'
        '    gauss_p = config["corruptions"]["gaussian_blur"][severity]\n'
        '    motion_p = config["corruptions"]["motion_blur"][severity]\n'
        '    return {\n'
        '        "low_light": [("CLAHE", apply_clahe), ("Retinex", apply_retinex)],\n'
        '        "gaussian_blur": [("Wiener", lambda img, s=gauss_p["sigma"], k=gauss_p["kernel_size"]:\n'
        '                           apply_wiener_deconv(img, sigma=s, kernel_size=k))],\n'
        '        "motion_blur": [("Wiener (Motion PSF)", lambda img, k=motion_p["kernel_size"]:\n'
        '                         apply_motion_wiener_deconv(img, kernel_size=k))],\n'
        '        "sensor_noise": [("NLM Denoise", apply_nlm_denoise)],\n'
        '        "fog_haze": [("Dehaze (Dark Channel)", apply_dark_channel_prior_dehaze)]\n'
        '    }'
    )
    
    if old_rescue_map in src:
        src = src.replace(old_rescue_map, new_rescue_map)
        return list(src)  # Will be re-joined
    return None

def fix_rescue_loop_call(cell_source_list):
    """
    Replace `RESCUE_MAP[ctype]` with `get_rescue_map(sev, config)[ctype]`
    in the rescue inference loop.
    """
    src = ''.join(cell_source_list)
    
    # Pattern 1: "if ctype in RESCUE_MAP" -> "if ctype in get_rescue_map(sev, config)"
    src = src.replace(
        'if ctype in RESCUE_MAP',
        'if ctype in get_rescue_map(sev, config)'
    )
    
    # Pattern 2: "for r_name, r_func in RESCUE_MAP[ctype]" 
    # -> "for r_name, r_func in get_rescue_map(sev, config)[ctype]"
    src = src.replace(
        'for r_name, r_func in RESCUE_MAP[ctype]',
        'for r_name, r_func in get_rescue_map(sev, config)[ctype]'
    )
    
    return list(src)

def fix_fog_seed(cell_source_list):
    """Add seed parameter to apply_fog and pass seed in apply_corruption."""
    src = ''.join(cell_source_list)
    
    # Fix the function definition
    old_fog = (
        'def apply_fog(image, fog_coef_lower, fog_coef_upper, alpha_coef=0.1):\n'
        '    t = A.RandomFog(fog_coef_range=(fog_coef_lower, fog_coef_upper),\n'
        '                    alpha_coef=alpha_coef, p=1.0)\n'
        '    return t(image=image)["image"]'
    )
    
    new_fog = (
        'def apply_fog(image, fog_coef_lower, fog_coef_upper, alpha_coef=0.1, seed=42):\n'
        '    """Apply synthetic fog/haze using Albumentations RandomFog with deterministic seed."""\n'
        '    import random as _random\n'
        '    _random.seed(seed)\n'
        '    np.random.seed(seed % (2**31))\n'
        '    t = A.RandomFog(fog_coef_range=(fog_coef_lower, fog_coef_upper),\n'
        '                    alpha_coef=alpha_coef, p=1.0)\n'
        '    return t(image=image)["image"]'
    )
    
    if old_fog in src:
        src = src.replace(old_fog, new_fog)
    
    # Fix the call in apply_corruption to pass seed
    old_call = 'elif ctype == "fog_haze": return apply_fog(image, params["fog_coef_lower"], params["fog_coef_upper"], params.get("alpha_coef", 0.1))'
    new_call = 'elif ctype == "fog_haze": return apply_fog(image, params["fog_coef_lower"], params["fog_coef_upper"], params.get("alpha_coef", 0.1), seed=seed)'
    
    if old_call in src:
        src = src.replace(old_call, new_call)
    
    return list(src)

def fix_sensor_noise_docstring(cell_source_list):
    """Add docstring to apply_sensor_noise documenting the salt-and-pepper component."""
    src = ''.join(cell_source_list)
    
    old_noise = 'def apply_sensor_noise(image, gauss_var, seed=42):\n    rng = np.random.default_rng(seed)'
    new_noise = (
        'def apply_sensor_noise(image, gauss_var, seed=42):\n'
        '    """Apply Gaussian noise + 5% salt-and-pepper impulse noise.\n'
        '    \n'
        '    Note: The salt-and-pepper component (sp_ratio=0.05) adds impulse noise\n'
        '    on top of the Gaussian noise. This means NLM denoising is suboptimal\n'
        '    for this corruption type; a median filter would be more appropriate\n'
        '    for the impulse component.\n'
        '    """\n'
        '    rng = np.random.default_rng(seed)'
    )
    
    if old_noise in src:
        src = src.replace(old_noise, new_noise)
    
    return list(src)

def fix_severe_skip_and_baseline(nb):
    """
    For clean notebooks (01, 03): remove the `if sev == "severe": continue` guard
    and add baseline row emission.
    """
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        
        # Remove the severe skip
        if 'if sev == "severe"' in src and 'continue' in src:
            # Remove the two lines: `if sev == "severe": \n    continue`
            lines = cell['source']
            new_lines = []
            skip_next = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if 'if sev == "severe"' in stripped:
                    skip_next = True
                    continue
                if skip_next and stripped == 'continue':
                    skip_next = False
                    continue
                skip_next = False
                new_lines.append(line)
            cell['source'] = new_lines
            print("    Removed severe skip")
        
        # Also fix the rescue guard: `sev != "severe"` → remove
        if 'sev != "severe"' in src:
            lines = cell['source']
            new_lines = []
            for line in lines:
                if 'sev != "severe"' in line:
                    # Replace with just `if ctype in RESCUE_MAP:` or get_rescue_map
                    line = line.replace(' and sev != "severe"', '')
                new_lines.append(line)
            cell['source'] = new_lines
            print("    Removed rescue severe guard")
    
    return nb


def apply_all_fixes_to_notebook(nb_path, is_clean_mvtec=False):
    """Apply all applicable fixes to a single notebook."""
    nb = load_notebook(nb_path)
    name = os.path.basename(nb_path)
    print(f"\nProcessing {name}...")
    
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        
        # Fix 1: RESCUE_MAP → get_rescue_map
        if 'RESCUE_MAP = {' in src:
            result = fix_rescue_map(cell['source'])
            if result is not None:
                cell['source'] = [''.join(result)]
                print("  ✓ Fixed RESCUE_MAP → get_rescue_map()")
        
        # Fix 1b: Update rescue loop calls
        if 'RESCUE_MAP[ctype]' in src or 'in RESCUE_MAP' in src:
            cell['source'] = [''.join(fix_rescue_loop_call(cell['source']))]
            print("  ✓ Fixed rescue loop calls")
        
        # Fix 2: Seed apply_fog
        if 'def apply_fog(' in src:
            cell['source'] = [''.join(fix_fog_seed(cell['source']))]
            print("  ✓ Seeded apply_fog()")
        
        # Fix 3: Document salt-and-pepper
        if 'def apply_sensor_noise(' in src and 'sp_ratio' in src:
            cell['source'] = [''.join(fix_sensor_noise_docstring(cell['source']))]
            print("  ✓ Added sensor noise docstring")
    
    # Fix 4: Clean notebook severe skip + baseline (01, 03 only)
    if is_clean_mvtec:
        nb = fix_severe_skip_and_baseline(nb)
    
    save_notebook(nb, nb_path)


def main():
    notebooks = [
        ('01_mvtec_patchcore_clean.ipynb', True),   # Clean MVTec — has severe skip
        ('02_mvtec_patchcore_augmented.ipynb', False),
        ('03_mvtec_padim_clean.ipynb', True),        # Clean MVTec — has severe skip
        ('04_mvtec_padim_augmented.ipynb', False),
        ('05_visa_patchcore_clean.ipynb', False),
        ('06_visa_patchcore_augmented.ipynb', False),
        ('07_visa_padim_clean.ipynb', False),
        ('08_visa_padim_augmented.ipynb', False),
    ]
    
    for nb_name, is_clean_mvtec in notebooks:
        nb_path = os.path.join(NOTEBOOKS_DIR, nb_name)
        if os.path.exists(nb_path):
            apply_all_fixes_to_notebook(nb_path, is_clean_mvtec)
        else:
            print(f"⚠ Not found: {nb_path}")
    
    print("\n" + "="*60)
    print("All notebook fixes applied successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
