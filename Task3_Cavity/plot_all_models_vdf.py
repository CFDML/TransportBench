"""
Batch script to generate microscopic VDF reconstruction for all 6 models
"""
import os
import subprocess
import sys

MODELS = ['unet', 'fno', 'deeponet', 'vit', 'ae', 'pt']

def main():
    print("="*70)
    print("Generating Microscopic VDF Reconstructions for All Models")
    print("="*70 + "\n")
    
    results = {}
    
    for model in MODELS:
        print(f"\n{'='*70}")
        print(f"Processing {model.upper()} Model")
        print(f"{'='*70}\n")
        
        cmd = [sys.executable, 'plot_micro_vdf.py', '--model', model]
        
        try:
            result = subprocess.run(cmd, check=True, cwd=os.path.dirname(__file__) or '.')
            print(f"✓ {model.upper()} completed successfully")
            results[model] = True
        except subprocess.CalledProcessError as e:
            print(f"✗ {model.upper()} failed with error code {e.returncode}")
            results[model] = False
        except Exception as e:
            print(f"✗ {model.upper()} failed: {str(e)}")
            results[model] = False
    
    # Summary
    print("\n" + "="*70)
    print("Generation Summary")
    print("="*70)
    for model, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        output_file = f"microscopic_vdf_{model}.png"
        print(f"{model.upper():12s} : {status:12s} -> {output_file}")
    
    successful = sum(results.values())
    print(f"\nTotal: {successful}/{len(MODELS)} models completed successfully")
    print("="*70)

if __name__ == "__main__":
    main()
