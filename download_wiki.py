import os
import requests
from tqdm import tqdm  # This gives us progress bars

def download_file(url, target_path):
    """Download a file with a progress bar"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(target_path):
        print(f"File already exists: {target_path}")
        return
    
    # Start download
    print(f"Downloading {url} to {target_path}")
    response = requests.get(url, stream=True)
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    # Download with progress bar
    with open(target_path, 'wb') as file, tqdm(
            desc=os.path.basename(target_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)
            
    print(f"Download complete: {target_path}")

if __name__ == "__main__":
    # For a starter project, let's use a smaller Wikipedia dataset
    # Full dataset would be: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    
    # Simple English Wikipedia is much smaller (good for experimentation)
    wiki_url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
    target_path = "data/simplewiki-latest.xml.bz2"
    
    download_file(wiki_url, target_path)
