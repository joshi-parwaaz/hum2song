"""
download_songs.py - Download songs for the hum2song project.

This script downloads songs from multiple sources:
1. Free music archives
2. Direct download links for royalty-free songs
3. Local file system import

The downloaded songs are saved to the data/songs_mp3 directory.
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path
from urllib.parse import urlparse
import logging
import time
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Default paths
DEFAULT_SONG_DIR = os.path.join(project_root, "data", "songs_mp3")

# Try importing optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available. Limited download functionality.")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm library not available. Progress bars disabled.")

# List of royalty-free music URLs that are safe to use
SAMPLE_URLS = [
    "https://filesamples.com/samples/audio/mp3/sample1.mp3",  # Generic sample 1
    "https://filesamples.com/samples/audio/mp3/sample2.mp3",  # Generic sample 2
    "https://filesamples.com/samples/audio/mp3/sample3.mp3",  # Generic sample 3
    "https://filesamples.com/samples/audio/mp3/sample4.mp3",  # Generic sample 4
    "https://storage.googleapis.com/kagglesdsdata/datasets/43415/75190/genres_original/blues/blues.00001.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210916%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210916T062236Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=18885c64c4dd8cbf0d1e3842b79ee9010f88dfa6324bfcf0c3d56d7d1c38e2bd80a6e3c8a25ac52a29fffc39a5ad7c8fcbfe9109b7f8761ea2361bc2ba54c9c47f9a5eb0a96f09e985a0b0be11cb1eec5a08aa9cd3c7e29a67d1830dd1c882c67d6624d1e55b55126a95e7416c3380a1d1eb83e70f59dd2d6f07ed39a2f12b3ac5b4e3db5b7ac9a8ea6a5d52fe037d2b5b9c7b23e26d83f2e2a58f2bd0c5ddb59c2d79bcea7eb2add723d68c1b9ac50439eafb65cb6bd01fd1afcb7e62095dc91d01b9a92afaa8aa4af14d798de0350d19a8af6a3b8c97f04f52fb1e26d2a7d12b4b12df58e6dfed69a3c9deaa37e5b7b6eb1f9cca6f6b2bcb2",  # Blues
    "https://storage.googleapis.com/kagglesdsdata/datasets/43415/75190/genres_original/classical/classical.00001.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210916%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210916T062237Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=40fb35e4ea9ec57be45f6c49276ce79a5000c4bdd3df00b81b7d177b87c64a5c979a3bb90069c8a4a40e6b02a0111ea39e18d27a07a10ab769b8d9e0f01d60705c92ac7858d77e43c8ca6b21ee5e2e08febe0e2fddf3c73acb9dbe08b9f1a8a96af81ae9dbbe58cf1fbc26aa2f2e2dbb70c5324e32c7e4192cdd7dcc54e25d76a64835ab7d18adb00ab6cb76e2ed3fdc8dbb1fa8e19e0d88df1e3fde28cd6df7a73f0e6ca0be6ba38ae9cc7fe4ab8cc3d22e86f4f0a4c33f7a67df9c2c7fc1af44fa1eb5d0359ee5bf8a50df0e01cef5055ca2d1dea4d3d8bba1922ac57e68b2c1a7a9bc3ecdb2b1f41a3d78071e1ddb7bb7f8c3fdeda9b",  # Classical
    "https://storage.googleapis.com/kagglesdsdata/datasets/43415/75190/genres_original/country/country.00001.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210916%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210916T062237Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=86e5c6b67e0a2aa50af607ba16c12c7d59a81fd0b02bee6e5bab0a3c53ad2df5eb4dbc3a9c4e3254ed52a1d4e6fe30b47af56c3af3cb0fd1fa87a3bdf03b7c0a96f86a66c2a0ca63a0c8553b72a1fd57c0db3ff3b5ac61d2cbca1b36e9c2ca4dccfe6a6d12fc4644d671c02f98ae6d90cab9a06fa1e9c86fb3b9ae6a5c1e2a5ad9de8252b22c1e7c30c3ab1147dea71e36abd1f77f03c5c6c3e44b97098a1adbe64fd22304fc6d1b5cef0a6deea7b3cd1e6b55ee95068d073e10f5213fc3a1b8f6bd9a0ec3f72a1aa5dae3e9eda9c0daf1e3126fc70f1db073d51f1c3e45f67f9ed7d4eb06f78a8afdbefeba6f4d3b2c67a6ae40c8a",  # Country
    "https://storage.googleapis.com/kagglesdsdata/datasets/43415/75190/genres_original/disco/disco.00001.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210916%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210916T062237Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=0da5afd9bd7968e7bdd0d7b1ef9b42ef29a00c7f6da9cb9c55a0c33845e5dddfedb1b978f9dd52fea3694c6c01c932f6b88a36b8ea46c822f7f0f6d8b6b52fa7a04bb21c2f2e5e2bb0a02cc07c0f76ff1b1ed1cf2f75bdbea9af34c0bb26f05b254ae15df9b69797d54dc11fba6b0cad41ce6f0a83c54dd8ec3b8429f1c67cdc0ae9b9d93eb9b8ed00dbe91ec01733dbf0a2f7c8ebdc83a2c9d9d41efb7d5e9caa3ffbcb0ad3f6f89e12be5dd21df1d4eed1c60f23eb14ae02b9e17dc01dd9a0a95cfd2f0ee55febc1c8c1c3daae2cf6de26e35daed3c4b54be5e54a9e7a3ff96d1bef03ed6cc7acee5dec3ebc2b7c0a6c0d9c8ae13",  # Disco
    "https://storage.googleapis.com/kagglesdsdata/datasets/43415/75190/genres_original/hiphop/hiphop.00001.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210916%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210916T062237Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=11097f38acd6315a77c7e2ee2e5c84b83c2a85c15aaa82f66cffb1c072ecdffe10d22db1d48c7d3bfebfe51cfc89b04664ad0f1adf2d67b3af55b38a2a1c0dfe3af9bdb7d53bc8d07c4b9fce2f39cd6c3b5e3cf284aedf13c2e00f5d4b1d881e56d0a8e44fc6a11e4f693ecfa43c5b5f6cdc2a5d1c4bae70b8ef3e0ab10613cca2f36a68ec00f6dda7e49773fbf810bb2c7ea30c9344fec2abc8b4fe5f62c7ae4ecef1a9c37be43a89c6a0b74b7c5e56f9d6bdfdfb94c4d3fd4c6b35fd62a5c2fa9af1d21f76dea27b4dc7ff9bc5d04e4fd8edb2c4b8d78f242bf4ad9b3f15ca75d4bf2cc42d58b44f15c6bdd09fafc7f7a30f7",  # Hip Hop
    "https://storage.googleapis.com/kagglesdsdata/datasets/43415/75190/genres_original/jazz/jazz.00001.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20210916%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210916T062237Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=5f2b9cc7ad10e6056a8295bcd1a856dea6dcceea77eb4a36dc0e8e6dcc81fdd1d20b7e9b6c95a04845d1ec98e89237bed81f0c8f83723d1b28ef47a0c3c1e57d05f0e73ddd98e1740c72b14fdb70bab20962edfba4eb9dbfad33ff4ae5cf9dccdac7ab0caae5b53ef60ce50b38a1e1cf3a53e6c05add1ec13d5bcc8bc34dfd0aec88c71fbb56e78a7bedd4f0b08d53e09e4339ab53ba5edea0b5c8fc25b0d63d14ca93a8c31f8af9f30e3d0c78f3c6ef48ef9a3f75f3972f0eb5bf7a18f1f7abfd11eb0fc5ae10abab9e36a19ec9ee1db49f7e2f88edd67de8c53c553b1a23e30d88eb01e4ec7f42caa1c6414b15dce6bd2d37ad3d8fd0",  # Jazz
]

def create_directories() -> None:
    """Create necessary directories if they don't exist."""
    os.makedirs(DEFAULT_SONG_DIR, exist_ok=True)

def download_file(url: str, save_path: str) -> Dict[str, Any]:
    """
    Download a file from a URL and save it to a path.
    
    Args:
        url: URL to download from
        save_path: Path to save the file to
        
    Returns:
        Dictionary with download result
    """
    if not REQUESTS_AVAILABLE:
        return {"success": False, "error": "requests library not available", "url": url}
    
    try:
        start_time = time.time()
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        file_size = int(response.headers.get('content-length', 0))
        
        # Extract filename from URL if not provided in path
        if os.path.isdir(save_path):
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = f"download_{int(time.time())}.mp3"
            save_path = os.path.join(save_path, filename)
        
        # Download with progress bar if tqdm is available
        if TQDM_AVAILABLE and file_size > 0:
            with open(save_path, 'wb') as f, tqdm(
                desc=os.path.basename(save_path),
                total=file_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        progress.update(len(chunk))
        else:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
        
        return {
            "success": True,
            "url": url,
            "save_path": save_path,
            "size": os.path.getsize(save_path),
            "elapsed_time": time.time() - start_time
        }
    
    except Exception as e:
        logger.error(f"Error downloading {url}: {str(e)}")
        return {
            "success": False,
            "url": url,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }

def download_sample_songs(save_dir: str = DEFAULT_SONG_DIR, max_songs: int = 10) -> Dict[str, Any]:
    """
    Download sample songs from the predefined list.
    
    Args:
        save_dir: Directory to save songs to
        max_songs: Maximum number of songs to download
        
    Returns:
        Dictionary with download results
    """
    start_time = time.time()
    
    # Limit the number of songs
    urls = SAMPLE_URLS[:max_songs]
    
    # Prepare save paths
    save_paths = []
    for i, url in enumerate(urls):
        filename = f"Sample-{i+1}.mp3"
        save_paths.append(os.path.join(save_dir, filename))
    
    # Download files in parallel
    successful = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {
            executor.submit(download_file, url, path): (url, path) 
            for url, path in zip(urls, save_paths)
        }
        
        for future in as_completed(future_to_url):
            url, path = future_to_url[future]
            try:
                result = future.result()
                if result["success"]:
                    successful.append(result)
                else:
                    failed.append(result)
            except Exception as e:
                failed.append({
                    "success": False,
                    "url": url,
                    "error": str(e)
                })
    
    # Return summary
    return {
        "success": len(failed) == 0,
        "successful": successful,
        "failed": failed,
        "elapsed_time": time.time() - start_time
    }

def import_local_files(source_dir: str, save_dir: str = DEFAULT_SONG_DIR, copy: bool = True) -> Dict[str, Any]:
    """
    Import audio files from a local directory.
    
    Args:
        source_dir: Directory to import from
        save_dir: Directory to save songs to
        copy: Whether to copy (True) or move (False) the files
        
    Returns:
        Dictionary with import results
    """
    import shutil
    
    start_time = time.time()
    
    if not os.path.exists(source_dir):
        return {
            "success": False,
            "error": f"Source directory {source_dir} does not exist",
            "elapsed_time": 0
        }
    
    # Find all MP3 files
    mp3_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))
    
    if not mp3_files:
        return {
            "success": False,
            "error": f"No MP3 files found in {source_dir}",
            "elapsed_time": 0
        }
    
    # Copy or move files
    successful = []
    failed = []
    
    for i, src_path in enumerate(mp3_files):
        filename = f"Local-{i+1}.mp3"
        dst_path = os.path.join(save_dir, filename)
        
        try:
            if copy:
                shutil.copy2(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)
            
            successful.append({
                "success": True,
                "source": src_path,
                "destination": dst_path,
                "size": os.path.getsize(dst_path)
            })
        
        except Exception as e:
            failed.append({
                "success": False,
                "source": src_path,
                "error": str(e)
            })
    
    return {
        "success": len(failed) == 0,
        "successful": successful,
        "failed": failed,
        "elapsed_time": time.time() - start_time
    }

def main():
    """Main function to download songs."""
    parser = argparse.ArgumentParser(description="Download songs for the hum2song project.")
    parser.add_argument("--output", "-o", default=DEFAULT_SONG_DIR,
                        help=f"Output directory for downloaded songs. Default: {DEFAULT_SONG_DIR}")
    parser.add_argument("--source", "-s", choices=["samples", "local"],
                        default="samples", help="Source to download from: samples (default) or local")
    parser.add_argument("--local-dir", "-l", help="Local directory to import MP3 files from")
    parser.add_argument("--max", "-m", type=int, default=10, help="Maximum number of songs to download")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying when using local source")
    
    args = parser.parse_args()
    
    # Create output directory
    create_directories()
    
    # Download or import songs
    if args.source == "samples":
        logger.info(f"Downloading {args.max} sample songs to {args.output}...")
        result = download_sample_songs(args.output, args.max)
    elif args.source == "local":
        if not args.local_dir:
            logger.error("Please specify a local directory with --local-dir")
            return 1
        logger.info(f"Importing MP3 files from {args.local_dir} to {args.output}...")
        result = import_local_files(args.local_dir, args.output, not args.move)
    else:
        logger.error(f"Unknown source: {args.source}")
        return 1
    
    # Print results
    if result["success"]:
        logger.info(f"Download complete. {len(result['successful'])} files downloaded successfully, {len(result['failed'])} files failed.")
    else:
        if "error" in result:
            logger.error(f"Download failed: {result['error']}")
        else:
            logger.error(f"Download partially failed. {len(result['successful'])} files downloaded successfully, {len(result['failed'])} files failed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
