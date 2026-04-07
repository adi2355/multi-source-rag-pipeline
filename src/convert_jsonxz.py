#!/usr/bin/env python
"""
Script to convert Instagram's JSON.XZ files to regular JSON files
This will:
1. Find all JSON.XZ files in the Instaloader directory
2. Decompress them and extract key metadata
3. Save the metadata as regular JSON files in the same format expected by transcriber.py
"""
import os
import sys
import json
import lzma
import logging
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# Import from our existing modules
from config import DATA_DIR, DOWNLOAD_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, 'logs', 'jsonxz_converter.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('jsonxz_converter')

# Base path for Instaloader data with Unicode slashes
BASE_UNICODE_PATH = '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads'

# Dictionary of special paths for accounts with Unicode slashes
SPECIAL_ACCOUNT_PATHS = {
    'rajistics': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕rajistics/rajistics',
    'studymlwithme': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕studymlwithme/studymlwithme',
    'techie007.dev': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕techie007.dev/techie007.dev',
    'theaiclubhouse': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕theaiclubhouse/theaiclubhouse',
    'thevarunmayya': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕thevarunmayya/thevarunmayya',
    'priyal.py': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕priyal.py/priyal.py',
    'parasmadan.in': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕parasmadan.in/parasmadan.in',
    'okaashish': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕okaashish/okaashish',
    'mar_antaya': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕mar_antaya/mar_antaya',
    'hamza_automates': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕hamza_automates/hamza_automates',
    'goyashy.ai': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕goyashy.ai/goyashy.ai',
    'edhonour': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕edhonour/edhonour',
    'the_a_i_club': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕the_a_i_club/the_a_i_club',
    'daily.ml.papers': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕daily.ml.papers/daily.ml.papers',
    'aws_peter': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕aws_peter/aws_peter',
    'agi.lambda': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕agi.lambda/agi.lambda',
    '100xengineers': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕100xengineers/100xengineers',
    '3blue1brown': '/home/adi235/MistralOCR/Instagram-Scraper/∕home∕adi235∕MistralOCR∕Instagram-Scraper∕data∕downloads∕3blue1brown/3blue1brown',   
    'digitals': '/home/adi235/MistralOCR/digitalsamaritan',
    'realrileybrown': '/home/adi235/MistralOCR/realrileybrown',
    'thevamshi_krishna': '/home/adi235/MistralOCR/thevamshi_krishna',
    'gkcs__': '/home/adi235/MistralOCR/gkcs__',
    
}

def setup_directories(account_name):
    """Create necessary directories if they don't exist"""
    metadata_dir = os.path.join(DOWNLOAD_DIR, account_name, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)
    return metadata_dir

def find_jsonxz_files(account_name):
    """Find all JSON.XZ files in the Instaloader directory for the given account"""
    jsonxz_files = []
    
    # Determine the correct path based on account name
    if account_name in SPECIAL_ACCOUNT_PATHS:
        account_path = SPECIAL_ACCOUNT_PATHS[account_name]
        logger.info(f"Using special path for {account_name}: {account_path}")
    else:
        # Try both standard path and Unicode path
        standard_path = os.path.join(DOWNLOAD_DIR, account_name)
        account_path = f"{BASE_UNICODE_PATH}/{account_name}/{account_name}"
        
        # Check if the account path with Unicode slashes exists
        test_cmd = f'[ -d "{account_path}" ] && echo "exists" || echo "not exists"'
        result = subprocess.run(test_cmd, shell=True, capture_output=True, text=True)
        if "not exists" in result.stdout:
            logger.info(f"Unicode path not found for {account_name}, using standard path: {standard_path}")
            account_path = standard_path
        else:
            logger.info(f"Using Unicode path for {account_name}: {account_path}")
    
    try:
        logger.info(f"Searching for JSON.XZ files in: {account_path}")
        
        # Escape characters in path for shell safety
        safe_path = account_path.replace('"', '\\"')
        
        # Use subprocess to find JSON.XZ files in the special path
        find_cmd = f'find "{safe_path}" -name "*.json.xz" 2>/dev/null || echo "find failed"'
        result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
        output = result.stdout.strip()
        
        if output and "find failed" not in output:
            files = output.split('\n')
            if files and files[0]:  # Check if there's at least one file
                logger.info(f"Found {len(files)} JSON.XZ files in {account_name}'s path")
                jsonxz_files.extend(files)
                return jsonxz_files
        
        # If first attempt failed, try using find with a specific pattern
        logger.info(f"First attempt failed, trying alternative approach to find files")
        
        # Try using ls instead to see if the directory exists and contains JSON.XZ files
        ls_cmd = f'ls -1 "{safe_path}"/*.json.xz 2>/dev/null || echo "ls failed"'
        ls_result = subprocess.run(ls_cmd, shell=True, capture_output=True, text=True)
        ls_output = ls_result.stdout.strip()
        
        if ls_output and "ls failed" not in ls_output:
            files = ls_output.split('\n')
            if files and files[0]:
                logger.info(f"Found {len(files)} JSON.XZ files using ls")
                jsonxz_files.extend(files)
                return jsonxz_files
        
        # If we still don't have files, try a more direct approach
        logger.warning(f"Standard methods failed, trying direct file listing")
        
        # Try using shell glob pattern directly
        glob_cmd = f'echo "{safe_path}"/*.json.xz'
        glob_result = subprocess.run(glob_cmd, shell=True, capture_output=True, text=True)
        glob_output = glob_result.stdout.strip()
        
        if glob_output and "*.json.xz" not in glob_output:  # Make sure pattern was expanded
            files = glob_output.split('\n')
            logger.info(f"Found {len(files)} JSON.XZ files using glob pattern")
            jsonxz_files.extend(files)
            
    except Exception as e:
        logger.error(f"Error finding JSON.XZ files: {str(e)}")
    
    return jsonxz_files

def find_all_accounts():
    """Find all accounts in the download directory"""
    accounts = []
    
    try:
        # First try the normal path
        for item in os.listdir(DOWNLOAD_DIR):
            item_path = os.path.join(DOWNLOAD_DIR, item)
            if os.path.isdir(item_path):
                accounts.append(item)
        
        # Then try the Unicode path
        result = subprocess.run(
            f'find "{BASE_UNICODE_PATH}" -maxdepth 1 -type d | grep -v "^{BASE_UNICODE_PATH}$"',
            shell=True, capture_output=True, text=True
        )
        if result.returncode == 0:
            dirs = result.stdout.strip().split('\n')
            for dir_path in dirs:
                if dir_path:
                    account = os.path.basename(dir_path)
                    if account not in accounts:
                        accounts.append(account)
    except Exception as e:
        logger.error(f"Error finding accounts: {str(e)}")
    
    return accounts

def normalize_video_path(video_path):
    """
    Convert the Unicode path to a regular path that can be used by the system
    """
    # Convert the Unicode slashes to regular slashes
    normalized_path = video_path.replace('∕', '/')
    
    # If the path has duplicate parts, simplify it
    if '/home/adi235/MistralOCR/Instagram-Scraper/home/adi235/MistralOCR/Instagram-Scraper/' in normalized_path:
        normalized_path = normalized_path.replace(
            '/home/adi235/MistralOCR/Instagram-Scraper/home/adi235/MistralOCR/Instagram-Scraper/',
            '/home/adi235/MistralOCR/Instagram-Scraper/'
        )
    
    # Fix specific path pattern for rajistics
    if '/downloads/rajistics/rajistics/' in normalized_path:
        # Ensure this is a real path that exists
        if not os.path.exists(normalized_path):
            # Try the direct path without normalization as a fallback
            if os.path.exists(video_path):
                logger.info(f"Using direct path without normalization: {video_path}")
                return video_path
    
    return normalized_path

def copy_video_file(video_path, account_name, shortcode):
    """
    Copy video file from the Instaloader path to the standard download directory
    Returns the new path where the video was copied to
    """
    try:
        # Normalize the source path
        normalized_source = normalize_video_path(video_path)
        
        # Make sure the source exists
        if not os.path.exists(normalized_source):
            logger.error(f"Source video does not exist: {normalized_source}")
            logger.info(f"Trying original path: {video_path}")
            
            # Try with the original path as a fallback
            if os.path.exists(video_path):
                normalized_source = video_path
            else:
                # Try to find the file by stripping parts of the path 
                # Sometimes paths have mixed separators that confuse os.path
                base_name = os.path.basename(video_path)
                # Try to find the file by searching for its basename
                find_cmd = f'find /home/adi235/MistralOCR/Instagram-Scraper -name "{base_name}" | head -1'
                result = subprocess.run(find_cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    found_path = result.stdout.strip()
                    logger.info(f"Found video using find command: {found_path}")
                    normalized_source = found_path
                else:
                    logger.error(f"Cannot find video file anywhere: {base_name}")
                    return None
        
        # Define the destination path in the standard download directory
        video_filename = os.path.basename(normalized_source)
        
        # If the filename is just a UTC timestamp, append the shortcode to make it more identifiable
        if "UTC" in video_filename and shortcode not in video_filename:
            basename, ext = os.path.splitext(video_filename)
            video_filename = f"{basename}_{shortcode}{ext}"
        
        destination_path = os.path.join(DOWNLOAD_DIR, account_name, video_filename)
        
        # Check if the file already exists at the destination
        if os.path.exists(destination_path):
            logger.info(f"Video already exists at destination: {destination_path}")
            return destination_path
        
        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Copy the file with subprocess for better handling of Unicode paths
        logger.info(f"Copying video from {normalized_source} to {destination_path}")
        
        # Use cp command via subprocess to handle potential Unicode issues
        result = subprocess.run(
            f'cp "{normalized_source}" "{destination_path}"',
            shell=True, capture_output=True, text=True
        )
        
        if result.returncode == 0:
            logger.info(f"Successfully copied video to: {destination_path}")
            return destination_path
        else:
            logger.error(f"Error copying video. Return code: {result.returncode}. Error: {result.stderr}")
            
            # Try more direct approach using Python's shutil if subprocess fails
            try:
                import shutil
                logger.info(f"Trying to copy using shutil: {normalized_source} to {destination_path}")
                shutil.copy2(normalized_source, destination_path)
                logger.info(f"Successfully copied using shutil: {destination_path}")
                return destination_path
            except Exception as e:
                logger.error(f"Error copying with shutil: {str(e)}")
                return None
            
    except Exception as e:
        logger.error(f"Error copying video file: {str(e)}")
        return None

def convert_jsonxz_to_json(jsonxz_file, metadata_dir, account_name, copy_videos=True):
    """
    Convert a single JSON.XZ file to a regular JSON file
    Extracts important metadata and saves it in the same format as downloader.py
    Also copies the corresponding video file to the standard download directory if copy_videos is True
    """
    try:
        logger.info(f"Processing: {jsonxz_file}")
        
        # Define output JSON path
        file_basename = os.path.basename(jsonxz_file).replace('.json.xz', '')
        
        # Get corresponding video file path
        video_path = jsonxz_file.replace('.json.xz', '.mp4')
        
        # Extract data from JSON.XZ
        with lzma.open(jsonxz_file, 'rt', encoding='utf-8') as f:
            try:
                insta_data = json.load(f)
                
                # Extract important metadata
                if 'node' in insta_data:
                    node = insta_data['node']
                    shortcode = node.get('shortcode')
                    
                    # If no shortcode found, use the filename
                    if not shortcode:
                        shortcode = f"unknown_{file_basename}"
                    
                    # Extract caption if available
                    caption = ""
                    if 'caption' in node and node['caption']:
                        caption = node['caption']
                    elif 'iphone_struct' in node and 'caption' in node['iphone_struct'] and node['iphone_struct']['caption']:
                        caption_data = node['iphone_struct']['caption']
                        if isinstance(caption_data, dict) and 'text' in caption_data:
                            caption = caption_data['text']
                    
                    # Extract date if available
                    date = None
                    if 'date' in node:
                        date = node['date']
                    elif 'taken_at' in node.get('iphone_struct', {}):
                        date = node['iphone_struct']['taken_at']
                    
                    # Normalize the video path to make it usable
                    normalized_video_path = normalize_video_path(video_path)
                    
                    # Copy the video file to the standard download directory if requested
                    final_video_path = normalized_video_path
                    if copy_videos:
                        new_video_path = copy_video_file(video_path, account_name, shortcode)
                        if new_video_path:
                            final_video_path = new_video_path
                    
                    # Create metadata object
                    metadata = {
                        'shortcode': shortcode,
                        'account': account_name,
                        'caption': caption,
                        'date': date,
                        'video_path': final_video_path,
                        'original_jsonxz_path': jsonxz_file,
                        'conversion_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Save as JSON file
                    output_path = os.path.join(metadata_dir, f"{shortcode}.json")
                    with open(output_path, 'w', encoding='utf-8') as out_f:
                        json.dump(metadata, out_f, ensure_ascii=False, indent=4)
                    
                    logger.info(f"Converted {jsonxz_file} to {output_path}")
                    return shortcode
                else:
                    logger.warning(f"No 'node' data found in {jsonxz_file}")
                    return None
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in {jsonxz_file}: {str(e)}")
                return None
    except Exception as e:
        logger.error(f"Error processing {jsonxz_file}: {str(e)}")
        return None

def main():
    """Main function to convert JSON.XZ files to JSON"""
    parser = argparse.ArgumentParser(description='Convert JSON.XZ files to regular JSON files')
    parser.add_argument('--account', type=str, help='Specific account to process')
    parser.add_argument('--all', action='store_true', help='Process all accounts')
    parser.add_argument('--force', action='store_true', help='Force reconversion of already converted files')
    parser.add_argument('--list-only', action='store_true', help='Only list files, don\'t convert')
    parser.add_argument('--list-accounts', action='store_true', help='List all available accounts')
    parser.add_argument('--verbose', action='store_true', help='Print additional debug information')
    parser.add_argument('--no-copy-videos', action='store_true', help='Skip copying video files to standard location')
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled")
    
    # List all accounts if requested
    if args.list_accounts:
        accounts = find_all_accounts()
        logger.info(f"Found {len(accounts)} accounts:")
        for account in accounts:
            logger.info(f" - {account}")
        return
    
    # Determine which accounts to process
    accounts_to_process = []
    if args.account:
        accounts_to_process = [args.account]
    elif args.all:
        accounts_to_process = find_all_accounts()
    else:
        logger.error("No account specified. Use --account ACCOUNT_NAME or --all to process all accounts")
        return
    
    # Process each account
    for account_name in accounts_to_process:
        logger.info(f"Processing account: {account_name}")
        
        # Create output directory
        metadata_dir = setup_directories(account_name)
        logger.info(f"Metadata directory for {account_name}: {metadata_dir}")
        
        # Find all JSON.XZ files for this account
        jsonxz_files = find_jsonxz_files(account_name)
        
        if not jsonxz_files:
            logger.error(f"No JSON.XZ files found for account: {account_name}")
            logger.info("Trying to list the directory contents to debug:")
            try:
                account_path = f"{BASE_UNICODE_PATH}/{account_name}/{account_name}"
                result = subprocess.run(
                    f'ls -la "{account_path}"',
                    shell=True, capture_output=True, text=True
                )
                logger.info(f"Directory contents:\n{result.stdout}")
            except Exception as e:
                logger.error(f"Error listing directory: {str(e)}")
            continue
        
        if args.list_only:
            logger.info(f"Files found for {account_name} (list-only mode):")
            for file in jsonxz_files:
                logger.info(f" - {file}")
            continue
        
        # Get existing JSON files
        existing_json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
        logger.info(f"Found {len(existing_json_files)} existing JSON files in metadata directory for {account_name}")
        
        # Process each JSON.XZ file
        converted_count = 0
        skipped_count = 0
        error_count = 0
        for i, jsonxz_file in enumerate(jsonxz_files):
            logger.info(f"Processing file {i+1}/{len(jsonxz_files)}: {jsonxz_file}")
            
            # Skip if already converted and not forcing
            file_basename = os.path.basename(jsonxz_file)
            if not args.force:
                # Try to extract shortcode from filename pattern
                parts = file_basename.replace('.json.xz', '').split('_')
                if len(parts) >= 3 and parts[2] != "UTC":
                    potential_shortcode = parts[2]
                    if f"{potential_shortcode}.json" in existing_json_files:
                        logger.info(f"Skipping {jsonxz_file} (already converted)")
                        skipped_count += 1
                        continue
            
            # Convert the file (and copy video if not disabled)
            try:
                result = convert_jsonxz_to_json(jsonxz_file, metadata_dir, account_name, not args.no_copy_videos)
                if result:
                    converted_count += 1
                else:
                    error_count += 1
                    logger.error(f"Failed to convert {jsonxz_file}")
            except Exception as e:
                error_count += 1
                logger.error(f"Error converting {jsonxz_file}: {str(e)}")
        
        logger.info(f"Conversion completed for {account_name}: {converted_count} files converted, {skipped_count} files skipped, {error_count} errors")
        logger.info(f"JSON files saved to: {metadata_dir}")
    
    logger.info("All accounts processed. You can now run transcriber.py on these JSON files")

if __name__ == "__main__":
    main() 