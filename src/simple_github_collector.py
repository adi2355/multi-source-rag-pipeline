#!/usr/bin/env python3
"""
Simplified GitHub repository collector for testing token-based access
"""

import os
import sys
import json
import time
import logging
import requests
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simple_github_collector')

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'github')
os.makedirs(DATA_DIR, exist_ok=True)

def create_github_session(token=None):
    """Create a GitHub API session with authentication if token is provided."""
    session = requests.Session()
    
    # Set user agent
    session.headers.update({
        'User-Agent': 'Simple-GitHub-Collector',
        'Accept': 'application/vnd.github.v3+json'
    })
    
    # Set token if provided
    if token:
        session.headers.update({
            'Authorization': f'token {token}'
        })
        logger.info("Using authenticated GitHub API requests")
    else:
        logger.warning("No GitHub API token found. Using unauthenticated requests (severe rate limiting will apply)")
    
    # Test the connection
    try:
        response = session.get('https://api.github.com/rate_limit')
        if response.status_code == 200:
            rate_info = response.json()['resources']['core']
            remaining = rate_info['remaining']
            reset_time = datetime.fromtimestamp(rate_info['reset']).strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"GitHub API connection successful. {remaining} requests remaining, resets at {reset_time}")
            return session
        else:
            logger.error(f"GitHub API connection failed with status: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error connecting to GitHub API: {str(e)}")
        return None

def get_repo_info(session, repo_name):
    """Get basic information about a repository."""
    logger.info(f"Getting info for repository: {repo_name}")
    try:
        response = session.get(f'https://api.github.com/repos/{repo_name}')
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error getting repository info. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception getting repository info: {str(e)}")
        return None

def get_repo_readme(session, repo_name):
    """Get the README content for a repository."""
    logger.info(f"Getting README for repository: {repo_name}")
    try:
        response = session.get(f'https://api.github.com/repos/{repo_name}/readme')
        if response.status_code == 200:
            data = response.json()
            # README is base64 encoded
            import base64
            readme_content = base64.b64decode(data['content']).decode('utf-8')
            return readme_content
        else:
            logger.warning(f"No README found for {repo_name}. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception getting README: {str(e)}")
        return None

def get_repo_content(session, repo_name, path=""):
    """Get the contents of a repository directory."""
    logger.info(f"Getting contents for repository: {repo_name}, path: {path or 'root'}")
    try:
        url = f'https://api.github.com/repos/{repo_name}/contents/{path}' if path else f'https://api.github.com/repos/{repo_name}/contents'
        response = session.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get contents. Status code: {response.status_code}")
            return []
    except Exception as e:
        logger.error(f"Exception getting contents: {str(e)}")
        return []

def get_file_content(session, file_url):
    """Get the content of a specific file."""
    try:
        response = session.get(file_url)
        if response.status_code == 200:
            data = response.json()
            # File content is base64 encoded
            import base64
            if data.get('encoding') == 'base64':
                return base64.b64decode(data['content']).decode('utf-8')
            return data.get('content', '')
        else:
            logger.warning(f"Failed to get file content. Status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Exception getting file content: {str(e)}")
        return None

def save_repo_data(repo_name, repo_info, readme, files):
    """Save repository data to the filesystem."""
    # Create directory for this repository
    safe_name = repo_name.replace('/', '_')
    repo_dir = os.path.join(DATA_DIR, safe_name)
    os.makedirs(repo_dir, exist_ok=True)
    
    # Save metadata
    with open(os.path.join(repo_dir, 'metadata.json'), 'w') as f:
        json.dump(repo_info, f, indent=2)
    logger.info(f"Saved metadata to {repo_dir}/metadata.json")
    
    # Save README
    if readme:
        with open(os.path.join(repo_dir, 'README.md'), 'w') as f:
            f.write(readme)
        logger.info(f"Saved README to {repo_dir}/README.md")
    
    # Save files
    files_dir = os.path.join(repo_dir, 'files')
    os.makedirs(files_dir, exist_ok=True)
    
    for file_info in files:
        if 'content' in file_info and file_info.get('type') == 'file':
            file_path = file_info['path']
            safe_path = file_path.replace('/', '_')
            
            with open(os.path.join(files_dir, safe_path), 'w') as f:
                f.write(file_info['content'])
            logger.info(f"Saved file {file_path} to {files_dir}/{safe_path}")
    
    return repo_dir

def collect_python_files(session, repo_name, max_files=10):
    """Collect Python and other valuable files from a repository."""
    collected_files = []
    
    # Get repository content
    contents = get_repo_content(session, repo_name)
    
    # Filter for Python files first
    python_files = [item for item in contents if item.get('type') == 'file' and 
                  item.get('name', '').endswith('.py')]
    
    # Also get some markdown and notebook files
    md_files = [item for item in contents if item.get('type') == 'file' and 
               (item.get('name', '').endswith('.md') or 
                item.get('name', '').endswith('.ipynb'))]
    
    # Combine files with Python files first
    files_to_get = python_files + md_files
    
    # Limit the number of files
    files_to_get = files_to_get[:max_files]
    
    # Get file contents
    for file_info in files_to_get:
        file_content = get_file_content(session, file_info['download_url'])
        if file_content:
            file_info['content'] = file_content
            collected_files.append(file_info)
            logger.info(f"Collected file: {file_info['path']}")
    
    return collected_files

def process_repo(repo_name, token=None, max_files=10):
    """Process a single repository."""
    # Create GitHub session
    session = create_github_session(token)
    if not session:
        logger.error("Failed to create GitHub session")
        return False
    
    # Get repository info
    repo_info = get_repo_info(session, repo_name)
    if not repo_info:
        logger.error(f"Failed to get repository info for {repo_name}")
        return False
    
    # Get README
    readme = get_repo_readme(session, repo_name)
    
    # Collect files
    files = collect_python_files(session, repo_name, max_files)
    
    # Save data
    repo_dir = save_repo_data(repo_name, repo_info, readme, files)
    logger.info(f"Repository data saved to {repo_dir}")
    
    return True

def process_repos_from_file(file_path, token=None, max_files=10):
    """Process multiple repositories from a text file."""
    if not os.path.isfile(file_path):
        logger.error(f"Repository file not found: {file_path}")
        return 0
    
    # Read repository names
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            repo_names = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    repo_names.append(line)
    except Exception as e:
        logger.error(f"Error reading repository file: {str(e)}")
        return 0
    
    logger.info(f"Found {len(repo_names)} repositories in file {file_path}")
    logger.info(f"Repositories to process: {', '.join(repo_names)}")
    
    # Process each repository
    success_count = 0
    for i, repo_name in enumerate(repo_names, 1):
        logger.info(f"Processing repository {i}/{len(repo_names)}: {repo_name}")
        
        if process_repo(repo_name, token, max_files):
            success_count += 1
        
        logger.info(f"Completed {i}/{len(repo_names)} repositories. Success: {success_count}")
        
        # Add a small delay between repositories
        if i < len(repo_names):
            time.sleep(2)
    
    logger.info(f"Repository batch processing complete. Successfully processed {success_count}/{len(repo_names)} repositories.")
    logger.info(f"Data stored in: {DATA_DIR}")
    return success_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simple GitHub repository collector')
    parser.add_argument('--repo', type=str, help='Specific repository to collect (format: owner/repo)')
    parser.add_argument('--repo-file', type=str, help='Path to text file containing repository names, one per line')
    parser.add_argument('--token', type=str, help='GitHub API token')
    parser.add_argument('--max-files', type=int, default=10, help='Maximum number of files to collect per repository')
    args = parser.parse_args()
    
    if args.repo_file:
        # Process repositories from file
        process_repos_from_file(args.repo_file, args.token, args.max_files)
    elif args.repo:
        # Process a single repository
        process_repo(args.repo, args.token, args.max_files)
    else:
        parser.print_help() 