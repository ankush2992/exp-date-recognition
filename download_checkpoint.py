import os
import gdown
import sys
from pathlib import Path

def download_checkpoints():
    """
    Downloads the model checkpoints from Google Drive
    Returns:
        bool: True if download was successful, False otherwise
    """
    # Create checkpoints directory if it doesn't exist
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Your Google Drive file ID
    file_id = "1PI4sChoFi7UCF0Ejijds_mbN6scVSSr9"
    
    # Output path
    output_path = checkpoint_dir / "model_epoch_final.pth"
    
    if output_path.exists():
        print("Checkpoint file already exists. Skipping download.")
        return True
    
    try:
        print("Downloading model checkpoint...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False)
        
        if output_path.exists() and os.path.getsize(output_path) > 0:
            print(f"Successfully downloaded checkpoint to {output_path}")
            return True
        else:
            print("Download failed or file is empty!")
            return False
            
    except Exception as e:
        print(f"Error downloading checkpoint: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_checkpoints()
    if not success:
        sys.exit(1)