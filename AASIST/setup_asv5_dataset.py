import os
import tarfile
import glob

dataset_root = r"C:\Users\HazCodes\Documents\Datasets\ASVspoof5"

def extract_all():
    print(f"Starting extraction at {dataset_root}...")
    
    protocol_tars = glob.glob(os.path.join(dataset_root, "ASVspoof5_protocols.tar*"))
    for protocol_tar in protocol_tars:
        print(f"Extracting {os.path.basename(protocol_tar)}...")
        mode = "r:gz" if protocol_tar.endswith(".gz") else "r"
        with tarfile.open(protocol_tar, mode) as tar:
            tar.extractall(path=dataset_root)
            
    print("Extracting training chunks...")
    t_tars = glob.glob(os.path.join(dataset_root, "flac_T_*.tar"))
    if not t_tars:
        print("  Warning: No flac_T_*.tar files found.")
    for tar_path in t_tars:
        print(f"  Extracting {os.path.basename(tar_path)}...")
        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=dataset_root)
        except Exception as e:
            print(f"  Error extracting {tar_path}: {e}")

    print("Extracting dev chunks...")
    d_tars = glob.glob(os.path.join(dataset_root, "flac_D_*.tar"))
    if not d_tars:
        print("  Warning: No flac_D_*.tar files found.")
    for tar_path in d_tars:
        print(f"  Extracting {os.path.basename(tar_path)}...")
        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=dataset_root)
        except Exception as e:
            print(f"  Error extracting {tar_path}: {e}")

    print("\nExtraction Complete!")
    print("Your folders 'flac_T', 'flac_D', and 'protocols' (or similar) should now be ready.")
    print(f"Found in {dataset_root}:")
    for d in os.listdir(dataset_root):
        if os.path.isdir(os.path.join(dataset_root, d)):
            print(f"  - {d}/")

if __name__ == "__main__":
    extract_all()