"""
Module for extracting audio from videos and transcribing with Whisper
with added batch processing support
"""
import os
import json
import glob
import subprocess
import logging
import signal
import sys
import concurrent.futures
from tqdm import tqdm
import whisper
import torch
import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional

from config import (
    DOWNLOAD_DIR,
    AUDIO_DIR,
    TRANSCRIPT_DIR,
    WHISPER_MODEL
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcriber.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('transcriber')

def setup_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(AUDIO_DIR, exist_ok=True)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    
    # Create account-specific directories
    for account_dir in os.listdir(DOWNLOAD_DIR):
        if os.path.isdir(os.path.join(DOWNLOAD_DIR, account_dir)):
            os.makedirs(os.path.join(AUDIO_DIR, account_dir), exist_ok=True)
            os.makedirs(os.path.join(TRANSCRIPT_DIR, account_dir), exist_ok=True)

def extract_audio(video_path, audio_path):
    """Extract audio from video using FFmpeg"""
    try:
        # First check if video has audio stream
        check_cmd = f'ffprobe -i "{video_path}" -show_streams -select_streams a -loglevel error'
        result = subprocess.run(check_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # If no audio stream is found, return False
        if not result.stdout.strip():
            logger.warning(f"No audio stream found in {video_path}, skipping extraction")
            return False
            
        # If we have an audio stream, proceed with extraction
        cmd = f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y'
        process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if extraction was successful
        if process.returncode != 0:
            logger.error(f"FFmpeg failed to extract audio from {video_path}: {process.stderr.decode('utf-8', errors='replace')}")
            return False
            
        # Make sure the output file exists and has a reasonable size
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            return True
        else:
            logger.error(f"Output audio file missing or empty for {video_path}")
            return False
    except Exception as e:
        logger.error(f"Error extracting audio from {video_path}: {str(e)}")
        return False

def extract_audio_batch(video_audio_pairs: List[Tuple[str, str]], max_workers: int = 4) -> List[str]:
    """
    Extract audio from multiple videos in parallel
    
    Args:
        video_audio_pairs: List of (video_path, audio_path) tuples
        max_workers: Maximum number of parallel extraction processes
        
    Returns:
        List of successfully processed audio paths
    """
    global shutdown_requested
    successful_audio_paths = []
    
    # Create a function to handle extraction with error handling
    def process_pair(pair):
        global shutdown_requested
        if shutdown_requested:
            return pair[1], False
        try:
            return pair[1], extract_audio(pair[0], pair[1])
        except Exception as e:
            logger.error(f"Unexpected error processing {pair[0]}: {str(e)}")
            return pair[1], False
    
    # Use ThreadPoolExecutor for parallel extraction
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_pair, pair) for pair in video_audio_pairs]
            
            # Process results as they complete
            for i, future in enumerate(tqdm(
                concurrent.futures.as_completed(futures), 
                total=len(futures),
                desc="Extracting audio in parallel"
            )):
                try:
                    audio_path, success = future.result()
                    if success:
                        successful_audio_paths.append(audio_path)
                except Exception as e:
                    logger.error(f"Error processing result: {str(e)}")
                
                # Check if shutdown was requested
                if shutdown_requested and i % 10 == 0:  # Check periodically
                    logger.info("Shutdown requested, waiting for current extractions to finish...")
                    break
    
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received during audio extraction. Attempting graceful shutdown...")
        shutdown_requested = True
    
    logger.info(f"Successfully extracted audio from {len(successful_audio_paths)}/{len(video_audio_pairs)} videos")
    return successful_audio_paths

def transcribe_batch(
    model, 
    audio_paths: List[str], 
    account_map: Dict[str, str], 
    base_name_map: Dict[str, str],
    metadata_map: Dict[str, Dict],
    batch_size: int = 16
) -> None:
    """
    Transcribe a batch of audio files
    
    Args:
        model: Loaded Whisper model
        audio_paths: List of audio file paths to transcribe
        account_map: Mapping of audio paths to account names
        base_name_map: Mapping of audio paths to base filenames
        metadata_map: Mapping of audio paths to metadata
        batch_size: Number of files to process in one batch
    """
    global shutdown_requested
    total_batches = (len(audio_paths)-1)//batch_size + 1
    
    for i in range(0, len(audio_paths), batch_size):
        if shutdown_requested:
            logger.info("Shutdown requested, stopping transcription after current batch")
            break
            
        batch = audio_paths[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{total_batches} with {len(batch)} files")
        
        for audio_path in tqdm(batch, desc=f"Transcribing batch {i//batch_size + 1}/{total_batches}"):
            try:
                if shutdown_requested:
                    logger.info("Shutdown requested, finishing current file...")
                    break
                    
                # Check if audio file exists and has content
                if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                    logger.warning(f"Skipping non-existent or empty audio file: {audio_path}")
                    continue
                
                # Extract information from maps
                account = account_map[audio_path]
                base_name = base_name_map[audio_path]
                metadata = metadata_map.get(audio_path, {})
                filename = f"{base_name}.mp4"
                
                # Define transcript path
                transcript_path = os.path.join(TRANSCRIPT_DIR, account, f"{base_name}.json")
                
                # Skip if already transcribed
                if os.path.exists(transcript_path):
                    logger.info(f"Skipping already transcribed file: {filename}")
                    continue
                
                # Transcribe audio
                result = model.transcribe(
                    audio_path,
                    beam_size=1,            # Reduce beam size (default is 5)
                    best_of=1,              # Don't generate multiple candidates
                    temperature=0.0,        # Use greedy decoding (no sampling)
                    fp16=False              # Use full precision (avoid half-precision errors)
                )
                
                # Create transcript with metadata
                transcript_data = {
                    "text": result["text"],
                    "segments": result["segments"],
                    "language": result["language"],
                    "filename": filename,
                    "account": account,
                    **metadata
                }
                
                # Save transcript
                os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
                with open(transcript_path, 'w', encoding='utf-8') as f:
                    json.dump(transcript_data, f, ensure_ascii=False, indent=4)
                
                logger.info(f"Transcription complete for {filename}")
                
                # Clear CUDA cache after each file to prevent memory build-up
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {str(e)}")
                # Continue with next file instead of failing the entire batch
                continue
        
        # Clear CUDA cache after each batch to prevent memory build-up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_metadata(download_dir: str, account: str, base_name: str) -> Dict:
    """Get metadata if available"""
    metadata_path = os.path.join(download_dir, account, "metadata", f"{base_name}.json")
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata from {metadata_path}: {str(e)}")
    return metadata

def process_videos(batch_size: int = 16, extraction_workers: int = 4, auto_batch_size: bool = True):
    """
    Process all downloaded videos that haven't been transcribed yet, using batch processing
    
    Args:
        batch_size: Number of audio files to transcribe in one batch
        extraction_workers: Number of parallel audio extraction processes
        auto_batch_size: Whether to automatically determine optimal batch size
    """
    global shutdown_requested
    setup_directories()
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU memory: {vram_gb:.2f} GB")
        
        # Auto-determine batch size if requested
        if auto_batch_size:
            batch_size = estimate_optimal_batch_size(vram_gb=vram_gb)
            logger.info(f"Auto-determined batch size: {batch_size}")
    
    # Load Whisper model with GPU acceleration if available
    logger.info(f"Loading Whisper model: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL).to(device)
    logger.info("Model loaded successfully")
    
    try:
        # Get all video files
        all_videos = []
        for account_dir in os.listdir(DOWNLOAD_DIR):
            account_path = os.path.join(DOWNLOAD_DIR, account_dir)
            if os.path.isdir(account_path):
                videos = glob.glob(os.path.join(account_path, "*.mp4"))
                all_videos.extend(videos)
        
        logger.info(f"Found {len(all_videos)} total videos to process")
        
        # Prepare for batch processing
        video_audio_pairs = []
        account_map = {}     # Maps audio_path -> account
        base_name_map = {}   # Maps audio_path -> base_name
        metadata_map = {}    # Maps audio_path -> metadata
        
        # Collect videos that need processing
        for video_path in all_videos:
            if shutdown_requested:
                logger.info("Shutdown requested, stopping video collection")
                break
                
            # Extract account name and filename
            parts = video_path.split(os.sep)
            account = parts[-2]
            filename = os.path.basename(video_path)
            base_name = os.path.splitext(filename)[0]
            
            # Define paths
            audio_path = os.path.join(AUDIO_DIR, account, f"{base_name}.wav")
            transcript_path = os.path.join(TRANSCRIPT_DIR, account, f"{base_name}.json")
            
            # Skip if already transcribed
            if os.path.exists(transcript_path):
                continue
            
            # Add to processing queue if audio extraction is needed
            if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
                os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                video_audio_pairs.append((video_path, audio_path))
            
            # Store mappings for later use regardless of whether we need to extract audio
            # This ensures we attempt transcription even for previously extracted audio
            account_map[audio_path] = account
            base_name_map[audio_path] = base_name
            metadata_map[audio_path] = get_metadata(DOWNLOAD_DIR, account, base_name)
        
        # Extract audio in parallel
        successful_audio_paths = []
        if video_audio_pairs:
            logger.info(f"Extracting audio for {len(video_audio_pairs)} videos")
            successful_audio_paths = extract_audio_batch(video_audio_pairs, max_workers=extraction_workers)
            logger.info(f"Successfully extracted {len(successful_audio_paths)} audio files")
        
        if shutdown_requested:
            logger.info("Shutdown requested, skipping transcription")
            return
        
        # Get all audio files that need transcription
        audio_to_transcribe = []
        
        # First add successfully extracted audio
        for audio_path in successful_audio_paths:
            if audio_path in account_map:  # Should always be true, but check to be safe
                audio_to_transcribe.append(audio_path)
        
        # Then add any pre-existing audio files that haven't been processed yet
        for audio_path in account_map.keys():
            if audio_path not in audio_to_transcribe and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                audio_to_transcribe.append(audio_path)
        
        # Transcribe in batches
        logger.info(f"Transcribing {len(audio_to_transcribe)} audio files in batches of {batch_size}")
        if audio_to_transcribe:
            transcribe_batch(
                model, 
                audio_to_transcribe, 
                account_map, 
                base_name_map, 
                metadata_map,
                batch_size=batch_size
            )
        
        if shutdown_requested:
            logger.info("Transcription process stopped due to shutdown request")
        else:
            logger.info("Transcription process completed successfully")
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Exiting gracefully...")
        shutdown_requested = True
    except Exception as e:
        logger.error(f"Unexpected error in process_videos: {str(e)}")
        logger.exception("Stack trace:")
    finally:
        # Clean up resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Process finished")

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully"""
    global shutdown_requested
    if shutdown_requested:
        logger.info("Forced shutdown requested. Exiting immediately.")
        sys.exit(1)
    else:
        logger.info("Graceful shutdown requested. Finishing current batch...")
        shutdown_requested = True
        
# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def estimate_optimal_batch_size(vram_gb=8, video_count=0, avg_duration=60):
    """
    Estimate optimal batch size based on available VRAM and video characteristics
    
    Args:
        vram_gb: Available VRAM in GB
        video_count: Number of videos to process
        avg_duration: Average video duration in seconds
        
    Returns:
        Estimated optimal batch size
    """
    # Base heuristic: ~200MB per minute of audio for Whisper processing
    vram_mb = vram_gb * 1024
    model_mb = {"tiny": 75, "base": 142, "small": 466, "medium": 1464, "large": 2952}
    whisper_model_size = WHISPER_MODEL
    
    # Reserve memory for the model
    available_mb = vram_mb - model_mb.get(whisper_model_size, 1500)
    
    # Reserve some overhead
    available_mb *= 0.8
    
    # Calculate MB needed per minute of audio
    mb_per_minute = 200
    
    # Convert avg_duration to minutes
    avg_minutes = avg_duration / 60
    
    # Calculate batch size based on available memory
    max_batch_size = int(available_mb / (mb_per_minute * avg_minutes))
    
    # Cap batch size to reasonable limits
    max_batch_size = max(4, min(max_batch_size, 32))
    
    # If video count is very small, adjust batch size accordingly
    if video_count > 0:
        max_batch_size = min(max_batch_size, video_count)
    
    logger.info(f"Estimated optimal batch size: {max_batch_size} for {vram_gb}GB VRAM")
    return max_batch_size