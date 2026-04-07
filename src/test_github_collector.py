#!/usr/bin/env python3
"""
Test script for the enhanced GitHub repository collector.
This script demonstrates how to use the collector with:
1. A specific repository query
2. A general search query for machine learning repositories
3. A text file containing multiple repository names for batch processing
"""

import os
import sys
import logging
import argparse
import config
from github_collector import (
    collect_github_repos, 
    create_github_session, 
    get_repo_info, 
    get_repo_readme,
    collect_valuable_files,
    process_file_content,
    chunk_repository_content,
    combine_repository_content,
    ensure_repo_fs_structure,
    save_repo_metadata,
    save_repo_readme,
    save_repo_file,
    save_repo_chunks,
    save_combined_content,
    assess_repository_value,
    get_db_connection,
    store_repo_in_db,
    store_repo_chunks,
    get_repo_directory_structure,
    collect_file_content,
    is_valuable_file,
    GITHUB_DATA_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'github_collector_test.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('test_github_collector')

def set_github_token(token):
    """Set GitHub API token in environment and config."""
    if token:
        logger.info("Setting GitHub API token")
        os.environ['GITHUB_TOKEN'] = token
        config.GITHUB_CONFIG['api_token'] = token
        return True
    return False

def test_specific_repo(repo_name, max_repos=1, fs_only=False):
    """Test collecting a specific GitHub repository by direct API access."""
    logger.info(f"Testing collection of specific repository: {repo_name}")
    
    # Create GitHub API session
    session = create_github_session()
    if not session:
        logger.error("Failed to create GitHub API session")
        return
    
    # Get database connection if needed
    conn = None
    if not fs_only:
        conn = get_db_connection()
        if not conn:
            logger.error("Failed to connect to database")
            if not fs_only:
                return
            else:
                logger.warning("Continuing with filesystem storage only")
    
    try:
        # Directly fetch repository info instead of using search
        repo_info = get_repo_info(session, repo_name)
        if not repo_info:
            logger.error(f"Failed to get info for repository: {repo_name}")
            return
        
        # Get repository README
        readme = get_repo_readme(session, repo_name)
        if readme:
            logger.info(f"Successfully retrieved README for {repo_name} ({len(readme)} characters)")
        else:
            logger.warning(f"No README found for {repo_name}")
        
        # Assess repository value
        value_assessment = assess_repository_value(repo_info, readme)
        value_score, value_category = value_assessment
        
        logger.info(f"Repository {repo_name}: Value score {value_score}/100 ({value_category})")
        
        # Determine how many files to collect based on repository value
        max_files = 30 if value_category == 'high' else (15 if value_category == 'medium' else 5)
        
        # Ensure we're collecting at least 10 files to improve the chances of getting code samples
        max_files = max(max_files, 10)
        
        # Collect valuable files with more aggressive approach
        logger.info(f"Collecting up to {max_files} valuable files from {repo_name}")
        files_data = collect_valuable_files(
            session,
            repo_name,
            max_files=max_files,
            include_readme=(readme is not None)
        )
        
        # If we didn't get enough files, try direct directory exploration
        if len(files_data) < 5:
            logger.info(f"Not enough files collected, attempting direct directory exploration")
            # Get repository directory structure
            dir_contents = get_repo_directory_structure(session, repo_name, recursive=True, max_depth=2)
            
            # Look specifically for Python files and examples
            for item in dir_contents:
                if len(files_data) >= max_files:
                    break
                    
                if item.get('type') != 'file':
                    continue
                    
                file_path = item.get('path', '')
                file_size = item.get('size', 0)
                
                if is_valuable_file(file_path, file_size) or file_path.endswith(('.py', '.ipynb')):
                    logger.info(f"Found additional file: {file_path}")
                    file_data = collect_file_content(session, repo_name, file_path, item)
                    if file_data:
                        files_data.append(file_data)
        
        logger.info(f"Collected {len(files_data)} valuable files from {repo_name}")
        for i, file_data in enumerate(files_data):
            logger.info(f"  {i+1}. {file_data.get('path')} ({file_data.get('size', 0)} bytes)")
        
        # Process files and create chunks
        for i, file_data in enumerate(files_data):
            if file_data.get('path', '').lower().endswith(('.py', '.ipynb', '.md', '.rst', '.txt')):
                files_data[i] = process_file_content(file_data)
        
        # Create file system structure and save content
        repo_path, docs_path, examples_path, processed_path = ensure_repo_fs_structure(repo_name)
        logger.info(f"Repository file system path: {repo_path}")
        
        # Save metadata
        metadata_path = save_repo_metadata(repo_info, repo_path)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save README
        if readme:
            readme_path = save_repo_readme(readme, repo_path)
            logger.info(f"Saved README to {readme_path}")
        
        # Save files - make sure this step succeeds
        saved_files = 0
        for file_data in files_data:
            file_path = save_repo_file(file_data, repo_name)
            if file_path:
                saved_files += 1
                logger.info(f"Saved file to {file_path}")
            else:
                logger.warning(f"Failed to save file {file_data.get('path')}")
        
        logger.info(f"Saved {saved_files}/{len(files_data)} files to {repo_path}")
        
        # Generate repository chunks for vector storage
        chunks = chunk_repository_content(
            repo_info,
            files_data,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Generate combined content for full-text search
        combined_content = combine_repository_content(
            repo_info,
            files_data,
            max_length=100000
        )
        
        # Save chunks
        if chunks:
            chunks_path = save_repo_chunks(chunks, repo_name)
            logger.info(f"Saved {len(chunks)} content chunks to {chunks_path}")
        
        # Save combined content
        if combined_content:
            combined_path = save_combined_content(combined_content, repo_name)
            logger.info(f"Saved combined content to {combined_path}")
        
        # Store in database if not fs_only
        if not fs_only and conn:
            # Add combined content as a special file
            if combined_content:
                files_data.append({
                    'path': '_combined_content.md',
                    'content': combined_content,
                    'processed_content': combined_content,
                    'file_type': 'documentation',
                    'is_readme': False
                })
                
            repo_id = store_repo_in_db(conn, repo_info, readme, files_data, value_assessment)
            
            if repo_id and chunks:
                # Store chunks for vector search
                store_repo_chunks(conn, repo_id, chunks)
            
            logger.info(f"Stored repository data in database with ID: {repo_id}")
        
        logger.info(f"Successfully processed repository: {repo_name}")
        return 1  # Return 1 repository processed
        
    except Exception as e:
        logger.error(f"Error processing repository {repo_name}: {e}")
        import traceback
        traceback.print_exc()
        return 0
        
    finally:
        if conn:
            conn.close()

def process_repo_file(file_path, fs_only=False):
    """Process multiple repositories from a text file.
    
    Args:
        file_path (str): Path to text file containing repository names, one per line.
        fs_only (bool): Whether to store only to filesystem, skip database operations.
        
    Returns:
        int: Number of repositories successfully processed.
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        logger.error(f"Repository file not found: {file_path}")
        return 0
    
    # Read repository names from file
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            repo_names = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    repo_names.append(line)
    except Exception as e:
        logger.error(f"Error reading repository file: {e}")
        return 0
    
    logger.info(f"Found {len(repo_names)} repositories in file {file_path}")
    logger.info(f"Repositories to process: {', '.join(repo_names)}")
    logger.info(f"Storage mode: {'Filesystem only' if fs_only else 'Filesystem and database'}")
    
    # Process each repository
    success_count = 0
    for i, repo_name in enumerate(repo_names, 1):
        logger.info(f"Processing repository {i}/{len(repo_names)}: {repo_name}")
        
        # Process repository
        result = test_specific_repo(repo_name, fs_only=fs_only)
        if result:
            success_count += 1
        
        logger.info(f"Completed {i}/{len(repo_names)} repositories. Success: {success_count}")
        
    logger.info(f"Repository batch processing complete. Successfully processed {success_count}/{len(repo_names)} repositories.")
    logger.info(f"Data stored in: {GITHUB_DATA_DIR}")
    return success_count

def test_general_search(search_query=None, max_repos=2, fs_only=False):
    """Test collecting repositories based on a general search query."""
    if search_query is None:
        search_query = config.GITHUB_CONFIG.get('search_query', 'machine learning language:python stars:>100')
        
    logger.info(f"Testing collection with general search query: {search_query}")
    logger.info(f"Storage mode: {'Filesystem only' if fs_only else 'Filesystem and database'}")
    
    # Save the original search query and fs_only setting
    original_query = config.GITHUB_CONFIG.get('search_query', '')
    original_fs_only = config.GITHUB_CONFIG.get('fs_only', False)
    
    try:
        # Set the search query and fs_only
        config.GITHUB_CONFIG['search_query'] = search_query
        config.GITHUB_CONFIG['fs_only'] = fs_only
        
        # Run the collector
        result = collect_github_repos(max_repos=max_repos)
        logger.info(f"Collection complete. Updated {result} repositories.")
        logger.info(f"Data stored in: {GITHUB_DATA_DIR}")
        
    finally:
        # Restore original settings
        config.GITHUB_CONFIG['search_query'] = original_query
        config.GITHUB_CONFIG['fs_only'] = original_fs_only

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the GitHub repository collector')
    parser.add_argument('--repo', type=str, help='Specific repository to collect (format: owner/repo)')
    parser.add_argument('--repo-file', type=str, help='Path to text file containing repository names, one per line')
    parser.add_argument('--query', type=str, help='General search query to use')
    parser.add_argument('--max-repos', type=int, default=2, help='Maximum number of repositories to collect')
    parser.add_argument('--token', type=str, help='GitHub API token to use (will override env var and config)')
    parser.add_argument('--fs-only', action='store_true', help='Store only to filesystem, skip database operations')
    args = parser.parse_args()
    
    # Set GitHub token if provided
    if args.token:
        set_github_token(args.token)
    
    if args.repo_file:
        # Process repositories from file
        repos_updated = process_repo_file(args.repo_file, fs_only=args.fs_only)
        logger.info(f"Batch processing complete. Updated {repos_updated} repositories.")
    elif args.repo:
        repos_updated = test_specific_repo(args.repo, max_repos=1, fs_only=args.fs_only)
        logger.info(f"Collection complete. Updated {repos_updated if repos_updated else 0} repositories.")
    elif args.query:
        test_general_search(args.query, max_repos=args.max_repos, fs_only=args.fs_only)
    else:
        # Run both tests
        print("\n===== Testing Specific Repository Collection =====")
        repos_updated = test_specific_repo('tensorflow/tensorflow', max_repos=1, fs_only=args.fs_only)
        logger.info(f"Collection complete. Updated {repos_updated if repos_updated else 0} repositories.")
        
        print("\n===== Testing General Search Query Collection =====")
        test_general_search(max_repos=args.max_repos, fs_only=args.fs_only) 