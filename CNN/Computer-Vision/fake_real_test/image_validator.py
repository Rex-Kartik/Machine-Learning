from PIL import Image
import shutil
import warnings
import os

# Disable PIL's decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)
# Suppress palette image warnings
warnings.filterwarnings('ignore', message='.*Palette.*')

dir = os.path.dirname(os.path.abspath(__file__))

def validate_images(directory):
  """Move corrupted images using TensorFlow decoder (sequential - no multiprocessing)"""
  # Collect all image paths
  image_paths = []
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_paths.append(os.path.join(root, file))
  
  print(f"\nFound {len(image_paths)} images. Validating...")
  
  # Sequential validation
  corrupted_count = 0
  for i, filepath in enumerate(image_paths, 1):
    if i % 100 == 0:
      print(f"  Progress: {i}/{len(image_paths)} checked...")
    
    try:
      img_data = tf.io.read_file(filepath)
      if filepath.lower().endswith(('.jpg', '.jpeg')):
        tf.image.decode_jpeg(img_data, channels=3)
      else:
        tf.image.decode_png(img_data, channels=3)
    except Exception as e:
      corrupted_count += 1
      subfolder = os.path.basename(os.path.dirname(filepath))
      corrupted_folder = os.path.join(os.path.dirname(os.path.dirname(filepath)), f"corrupted_{subfolder}")
      os.makedirs(corrupted_folder, exist_ok=True)
      dest_path = os.path.join(corrupted_folder, os.path.basename(filepath))
      try:
        shutil.move(filepath, dest_path)
        print(f"  ✓ MOVED corrupted file #{corrupted_count}: {os.path.basename(filepath)}")
      except Exception as move_err:
        print(f"  ✗ ERROR moving {os.path.basename(filepath)}: {move_err}")
  
  print(f"\n✓ Validation complete!")
  print(f"  Total images checked: {len(image_paths)}")
  print(f"  Corrupted images moved: {corrupted_count}")
  print(f"  Clean images for training: {len(image_paths) - corrupted_count}\n")

# Clean corrupted images first
validate_images(r'D:\Movies\archiv\archiv\archive\train')


# Clean corrupted images first using parallel TensorFlow validation
print("Validating images with TensorFlow's decoder (parallel processing)...")
validate_images(r'D:\Movies\archiv\archiv\archive\train')