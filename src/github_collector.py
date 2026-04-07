"""
GitHub Repository Collector Module
Collects and processes AI/ML repositories from GitHub for the knowledge base
"""
import os
import json
import logging
import time
import base64
import sqlite3
import requests
from datetime import datetime, timedelta
import config

# Configure logging
log_dir = os.path.join(config.DATA_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'github_collector.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('github_collector')

# List of repositories to collect (used as fallback if search doesn't yield enough results)
GITHUB_REPOS = [
    # Foundational ML/DL libraries
    'tensorflow/tensorflow',
    'pytorch/pytorch',
    'scikit-learn/scikit-learn',
    'huggingface/transformers',
    'keras-team/keras',
    
    # LLM & generative AI projects
    'openai/whisper',
    'facebookresearch/llama',
    'anthropics/claude-api',
    'google/gemma',
    'mistralai/mistral-src',
    
    # Training & infrastructure tools
    'ray-project/ray',
    'microsoft/DeepSpeed',
    'google/jax',
    
    # Research implementations
    'facebookresearch/fairseq',
    'openai/CLIP',
    'LAION-AI/Open-Assistant',
    
    # Learning resources
    'datawhalechina/pumpkin-book',
    'afshinea/stanford-cs-229-machine-learning',
    'microsoft/ML-For-Beginners'
]

def get_github_session():
    """Create a requests session with GitHub API token if available"""
    session = requests.Session()
    
    # Add GitHub API token if available
    github_token = config.GITHUB_CONFIG.get('api_token', '')
    if github_token:
        session.headers.update({
            'Authorization': f'token {github_token}',
            'Accept': 'application/vnd.github.v3+json'
        })
    else:
        session.headers.update({
            'Accept': 'application/vnd.github.v3+json'
        })
    
    # Set user agent
    session.headers.update({
        'User-Agent': 'AI-Knowledge-Base-Collector/1.0'
    })
    
    return session

def get_rate_limit_info(session):
    """Get the current GitHub API rate limit information"""
    try:
        response = session.get('https://api.github.com/rate_limit')
        if response.status_code == 200:
            data = response.json()
            core = data.get('resources', {}).get('core', {})
            remaining = core.get('remaining', 0)
            reset_timestamp = core.get('reset', 0)
            reset_time = datetime.fromtimestamp(reset_timestamp)
            
            logger.info(f"GitHub API rate limit: {remaining} requests remaining, resets at {reset_time}")
            return remaining, reset_time
        else:
            logger.warning(f"Failed to get rate limit info: {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Error getting rate limit info: {str(e)}")
        return None, None

def wait_for_rate_limit(session):
    """Check rate limit and wait if necessary"""
    remaining, reset_time = get_rate_limit_info(session)
    
    if remaining is None or reset_time is None:
        # If we can't get rate limit info, wait a conservative amount of time
        logger.warning("Could not get rate limit info, waiting 60 seconds")
        time.sleep(60)
        return
    
    if remaining <= 10:  # Keep a small buffer
        now = datetime.now()
        wait_seconds = (reset_time - now).total_seconds() + 5  # Add a small buffer
        
        if wait_seconds > 0:
            logger.warning(f"Rate limit almost reached. Waiting {wait_seconds:.1f} seconds until reset")
            time.sleep(wait_seconds)
        else:
            # If the reset time is in the past, wait a bit anyway
            logger.warning("Rate limit reset time is in the past, waiting 10 seconds as precaution")
            time.sleep(10)

def search_github_repos(session, topics, min_stars=1000, per_page=30, max_results=100):
    """Search for repositories based on topics and minimum stars"""
    repos = []
    topic_query = ' '.join([f'topic:{topic}' for topic in topics])
    query = f"{topic_query} stars:>={min_stars}"
    
    logger.info(f"Searching GitHub with query: {query}")
    
    page = 1
    while len(repos) < max_results:
        wait_for_rate_limit(session)
        
        try:
            response = session.get(
                'https://api.github.com/search/repositories',
                params={
                    'q': query,
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': per_page,
                    'page': page
                }
            )
            
            if response.status_code != 200:
                logger.error(f"GitHub search API returned error: {response.status_code} - {response.text}")
                break
                
            data = response.json()
            items = data.get('items', [])
            
            if not items:
                break
                
            repos.extend(items)
            total_count = data.get('total_count', 0)
            logger.info(f"Found {len(repos)}/{total_count} repositories (page {page})")
            
            if len(repos) >= max_results:
                logger.info(f"Reached maximum results limit of {max_results}")
                break
                
            # Check if we've reached the last page
            if len(items) < per_page:
                break
                
            page += 1
            
            # Add a small delay between requests
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Error searching GitHub repositories: {str(e)}")
            break
    
    return repos[:max_results]

def get_repo_info(session, repo_full_name):
    """Get detailed information about a repository"""
    wait_for_rate_limit(session)
    
    try:
        logger.info(f"Getting info for repository: {repo_full_name}")
        response = session.get(f'https://api.github.com/repos/{repo_full_name}')
        
        if response.status_code != 200:
            logger.error(f"GitHub API returned error for {repo_full_name}: {response.status_code} - {response.text}")
            return None
            
        return response.json()
    except Exception as e:
        logger.error(f"Error getting repository info for {repo_full_name}: {str(e)}")
        return None

def get_repo_readme(session, repo_full_name):
    """Get repository README content"""
    wait_for_rate_limit(session)
    
    try:
        logger.info(f"Getting README for repository: {repo_full_name}")
        response = session.get(f'https://api.github.com/repos/{repo_full_name}/readme')
        
        if response.status_code != 200:
            logger.warning(f"Could not find README for {repo_full_name}: {response.status_code}")
            return None
            
        data = response.json()
        content = data.get('content', '')
        encoding = data.get('encoding', 'base64')
        
        if content and encoding == 'base64':
            readme_content = base64.b64decode(content).decode('utf-8', errors='replace')
            
            # Limit readme length if necessary
            max_length = config.GITHUB_CONFIG.get('readme_max_length', 100000)
            if len(readme_content) > max_length:
                readme_content = readme_content[:max_length] + "... [truncated]"
                
            return readme_content
        
        return None
    except Exception as e:
        logger.error(f"Error getting README for {repo_full_name}: {str(e)}")
        return None

def should_update_repo(conn, repo_full_name):
    """Check if a repository should be updated based on last crawl time"""
    cursor = conn.cursor()
    
    try:
        # Check if the repo exists in the database
        cursor.execute("SELECT last_crawled FROM github_repos WHERE full_name = ?", (repo_full_name,))
        result = cursor.fetchone()
        
        if not result:
            # Repository not in database, should add it
            return True
            
        last_crawled = result[0]
        if not last_crawled:
            return True
            
        # Convert string to datetime
        last_crawled_dt = datetime.fromisoformat(last_crawled.replace('Z', '+00:00'))
        
        # Check if it's time to update
        update_frequency_days = config.GITHUB_CONFIG.get('update_frequency_days', 7)
        update_threshold = datetime.now() - timedelta(days=update_frequency_days)
        
        return last_crawled_dt < update_threshold
        
    except Exception as e:
        logger.error(f"Error checking if repo {repo_full_name} should be updated: {str(e)}")
        return True

def store_repo_in_db(conn, repo_data, readme):
    """Store repository information in the database"""
    cursor = conn.cursor()
    
    try:
        # Extract fields to store
        repo_full_name = repo_data.get('full_name')
        
        # Check if repo exists in database
        cursor.execute("SELECT id FROM github_repos WHERE full_name = ?", (repo_full_name,))
        existing_repo = cursor.fetchone()
        
        # Prepare data for insertion/update
        repo_values = {
            'id': repo_data.get('id'),
            'name': repo_data.get('name'),
            'full_name': repo_full_name,
            'description': repo_data.get('description') or '',
            'url': repo_data.get('html_url'),
            'stars': repo_data.get('stargazers_count', 0),
            'watchers': repo_data.get('watchers_count', 0),
            'forks': repo_data.get('forks_count', 0),
            'language': repo_data.get('language') or '',
            'last_push': repo_data.get('pushed_at'),
            'created_at': repo_data.get('created_at'),
            'updated_at': repo_data.get('updated_at'),
            'topics': json.dumps(repo_data.get('topics', [])),
            'readme': readme or '',
            'last_crawled': datetime.now().isoformat()
        }
        
        if existing_repo:
            # Update existing repo
            placeholders = ', '.join([f"{key} = ?" for key in repo_values.keys()])
            query = f"UPDATE github_repos SET {placeholders} WHERE full_name = ?"
            values = list(repo_values.values()) + [repo_full_name]
            cursor.execute(query, values)
            logger.info(f"Updated repository {repo_full_name} in database")
            
            # Get the repo ID for ai_content relation
            repo_id = existing_repo[0]
        else:
            # Insert new repo
            placeholders = ', '.join(['?'] * len(repo_values))
            columns = ', '.join(repo_values.keys())
            query = f"INSERT INTO github_repos ({columns}) VALUES ({placeholders})"
            cursor.execute(query, list(repo_values.values()))
            logger.info(f"Added new repository {repo_full_name} to database")
            
            # Get the inserted repo ID
            repo_id = cursor.lastrowid
        
        # Add entry to ai_content table
        content_values = {
            'title': repo_data.get('name'),
            'description': repo_data.get('description') or '',
            'content': readme or '',
            'source_type_id': 2,  # 2 = GitHub
            'source_id': str(repo_id),
            'url': repo_data.get('html_url'),
            'date_created': repo_data.get('created_at'),
            'date_collected': datetime.now().isoformat(),
            'metadata': json.dumps({
                'stars': repo_data.get('stargazers_count', 0),
                'language': repo_data.get('language') or '',
                'topics': repo_data.get('topics', []),
                'forks': repo_data.get('forks_count', 0)
            })
        }
        
        # Check if content already exists
        cursor.execute(
            "SELECT id FROM ai_content WHERE source_type_id = 2 AND source_id = ?", 
            (str(repo_id),)
        )
        existing_content = cursor.fetchone()
        
        if existing_content:
            # Update existing content
            placeholders = ', '.join([f"{key} = ?" for key in content_values.keys()])
            query = f"UPDATE ai_content SET {placeholders} WHERE source_type_id = 2 AND source_id = ?"
            values = list(content_values.values()) + [str(repo_id)]
            cursor.execute(query, values)
        else:
            # Insert new content
            placeholders = ', '.join(['?'] * len(content_values))
            columns = ', '.join(content_values.keys())
            query = f"INSERT INTO ai_content ({columns}) VALUES ({placeholders})"
            cursor.execute(query, list(content_values.values()))
        
        conn.commit()
        return True
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing repository {repo_data.get('full_name')} in database: {str(e)}")
        return False

def collect_github_repos(max_repos=None):
    """
    Main function to collect GitHub repositories
    Returns the number of successfully processed repositories
    """
    # Check if GitHub collection is enabled
    if not config.CONTENT_SOURCES.get('github', {}).get('enabled', False):
        logger.info("GitHub collection is disabled in configuration")
        return 0
    
    # Create necessary directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.DATA_DIR, 'logs'), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(config.DB_PATH)
    
    # Initialize session
    session = get_github_session()
    
    # Set default max_repos if not specified
    if max_repos is None:
        max_repos = config.CONTENT_SOURCES.get('github', {}).get('max_repos_per_run', 10)
    
    success_count = 0
    
    try:
        # Check rate limits first
        remaining, reset_time = get_rate_limit_info(session)
        if remaining is not None and remaining < max_repos * 3:  # Each repo might need multiple API calls
            logger.warning(f"Only {remaining} API requests remaining before rate limit reset, which might not be enough")
        
        # Get repositories to process
        repos_to_process = []
        
        # First try to use the search API
        topics = config.CONTENT_SOURCES.get('github', {}).get('topics', ['machine-learning', 'deep-learning'])
        min_stars = config.CONTENT_SOURCES.get('github', {}).get('repo_stars_minimum', 1000)
        
        search_results = search_github_repos(
            session=session,
            topics=topics,
            min_stars=min_stars,
            max_results=max_repos
        )
        
        # Extract repo full names from search results
        for repo in search_results:
            if repo.get('full_name'):
                repos_to_process.append(repo.get('full_name'))
        
        # If we didn't get enough repos from search, add from the predefined list
        if len(repos_to_process) < max_repos:
            remaining_count = max_repos - len(repos_to_process)
            for repo in GITHUB_REPOS:
                if repo not in repos_to_process and len(repos_to_process) < max_repos:
                    repos_to_process.append(repo)
        
        # Process repositories
        logger.info(f"Processing {len(repos_to_process)} repositories")
        
        for i, repo_full_name in enumerate(repos_to_process):
            logger.info(f"Processing repository {i+1}/{len(repos_to_process)}: {repo_full_name}")
            
            # Check if we need to update this repository
            if not should_update_repo(conn, repo_full_name):
                logger.info(f"Skipping {repo_full_name} - recently updated")
                continue
            
            # Get repository information
            repo_data = get_repo_info(session, repo_full_name)
            if not repo_data:
                logger.warning(f"Could not get information for {repo_full_name}, skipping")
                continue
            
            # Get repository README
            readme = get_repo_readme(session, repo_full_name)
            
            # Store in database
            if store_repo_in_db(conn, repo_data, readme):
                success_count += 1
            
            # Add a delay between repositories
            time.sleep(2)
            
            # Check if we've reached the maximum
            if success_count >= max_repos:
                logger.info(f"Reached maximum repository limit of {max_repos}")
                break
        
        logger.info(f"GitHub collection completed. Successfully processed {success_count}/{len(repos_to_process)} repositories")
        
    except Exception as e:
        logger.error(f"Error in GitHub collection process: {str(e)}")
    
    finally:
        # Close database connection
        conn.close()
    
    return success_count

if __name__ == "__main__":
    collect_github_repos() 