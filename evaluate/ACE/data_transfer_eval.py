import os
import shutil
import argparse
import subprocess

def copy_images(root_dir, out_dir):
    output_dirs = ["original", "adversarial"]
    for folder in output_dirs:
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        print("subdir_path:", subdir_path)

        if os.path.isdir(subdir_path) and subdir.startswith("sample_"):
            new_name = subdir

            file_mapping = {
                "original.png": os.path.join(out_dir, "original", f"{new_name}.png"),
                "adversarial.png": os.path.join(out_dir, "adversarial", f"{new_name}.png"),
            }

            for filename, dest_path in file_mapping.items():
                source_path = os.path.join(subdir_path, filename)
                print("source_path:", source_path)
                print("dest_path:", dest_path)

                if os.path.exists(source_path):
                    shutil.copy2(source_path, dest_path)
                    print(f"Y: Copied {source_path} -> {dest_path}")
                else:
                    print(f"N: File not found: {source_path}")

    print("Done copying files to output directories.")

def download_models_for_evaluate(out_dir):
    # First model
    models_dir = os.path.join(out_dir, 'ACE/models')
    os.chdir(models_dir)
    subprocess.run(['gdown', '--fuzzy',
                    'https://drive.google.com/uc?id=1xX_eExZKY679WQMC6uA-alfkUaoDBkIV'])
    os.chdir(out_dir)

    # Second model
    pretrained_models_dir = os.path.join(out_dir, 'ACE/pretrained_models')
    os.chdir(pretrained_models_dir)
    subprocess.run(['gdown', '--fuzzy', 
                    'https://drive.google.com/uc?id=1ScPPxFvggpOSKvlorFxMSULQFJbGGaAA'])
    os.chdir(out_dir)

    # Third model
    os.chdir(pretrained_models_dir)
    subprocess.run(['gdown', '--fuzzy', 
                    'https://drive.google.com/uc?id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU'])
    os.chdir(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy original and adversarial images into output folders.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to the root directory containing sample_* folders")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory to save copied images")

    args = parser.parse_args()
    copy_images(args.root_dir, args.out_dir)
    download_models_for_evaluate(args.out_dir)