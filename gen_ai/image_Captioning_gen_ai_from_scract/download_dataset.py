import kagglehub
import os
import shutil

DATASET = "adityajn105/flickr8k"
TARGET_DIR = "data/flickr8k"

def download_flickr8k():
    print(f"Downloading {DATASET} using kagglehub...")
    downloaded_path = kagglehub.dataset_download(DATASET)
    print(f"Downloaded to: {downloaded_path}")

    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
    
    for item in os.listdir(downloaded_path):
        src = os.path.join(downloaded_path, item)
        dst = os.path.join(TARGET_DIR, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    
    print(f"Dataset copied to {TARGET_DIR}")

    images_dir = os.path.join(TARGET_DIR, "Images")
    captions_file = os.path.join(TARGET_DIR, "captions.txt")
    if os.path.exists(images_dir) and os.path.exists(captions_file):
        print("✅ Verification successful: Images/ and captions.txt found.")
    else:
        print("❌ Verification failed. Please check the download.")

if __name__ == "__main__":
    download_flickr8k()