#!/usr/bin/env python3
"""
Script to parse Labelbox NDJSON annotations and create a training dataset with character integration.

UPDATED: 
- Character images are square-cropped from top
- Characters are stitched in grid layouts (1x2, 1x3, 2x2, 3+2, 3x2)
- Context frames are separate images (max 3)
- No colored borders (removed due to bleeding issues)

Requirements:
    ffmpeg (command-line tool)
    pip install beautifulsoup4 requests opencv-python numpy tqdm
"""

import json
import os
import logging
import requests
import csv
from pathlib import Path
import cv2
import numpy as np
import tempfile
from urllib.parse import urljoin
import re
import time
import subprocess
from typing import Dict, List, Tuple, Optional, Set
from bs4 import BeautifulSoup

# --------------------------- Logging ---------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOG = logging.getLogger("dataset_builder")

# ----------------------- Anime → Characters URL ----------------------

ANIME_MAPPING: Dict[str, str] = {
    "A_Couple_of_Cuckoos": "https://www.anime-planet.com/anime/a-couple-of-cuckoos/characters",
    "Can_a_Boy_Girl_Friendship_Survive": "https://www.anime-planet.com/anime/can-a-boy-girl-friendship-survive/characters",
    "Classroom_of_the_Elite": "https://www.anime-planet.com/anime/classroom-of-the-elite/characters",
    "Clevatess": "https://www.anime-planet.com/anime/clevatess/characters",
    "Food_for_the_Soul": "https://www.anime-planet.com/anime/food-for-the-soul/characters",
    "Frieren_Beyond_Journeys_End": "https://www.anime-planet.com/anime/frieren-beyond-journeys-end/characters",
    "I_Left_My_A_Rank_Party_to_Help_My_Former_Students_Reach_the_Dungeon_Depths": "https://www.anime-planet.com/anime/i-left-my-a-rank-party-to-help-my-former-students-reach-the-dungeon-depths/characters",
    "Lycoris_Recoil": "https://www.anime-planet.com/anime/lycoris-recoil/characters",
    "More_Than_a_Married_Couple_But_Not_Lovers": "https://www.anime-planet.com/anime/more-than-a-married-couple-but-not-lovers/characters",
    "My_Dress_Up_Darling": "https://www.anime-planet.com/anime/my-dress-up-darling/characters",
    "Once_Upon_a_Witchs_Death": "https://www.anime-planet.com/anime/once-upon-a-witchs-death/characters",
    "Rascal_Does_Not_Dream_of_Bunny_Girl_Senpai": "https://www.anime-planet.com/anime/rascal-does-not-dream-of-bunny-girl-senpai/characters",
    "Scooped_Up_by_an_S_Rank_Adventurer": "https://www.anime-planet.com/anime/scooped-up-by-an-s-rank-adventurer/characters",
    "Secrets_of_the_Silent_Witch": "https://www.anime-planet.com/anime/secrets-of-the-silent-witch/characters",
    "See_You_Tomorrow_at_the_Food_Court": "https://www.anime-planet.com/anime/see-you-tomorrow-at-the-food-court/characters",
    "Solo_Camping_For_Two": "https://www.anime-planet.com/anime/solo-camping-for-two/characters",
    "Summer_Pockets": "https://www.anime-planet.com/anime/summer-pockets/characters",
    "The_Apothecary_Diaries": "https://www.anime-planet.com/anime/the-apothecary-diaries/characters",
    "The_Brilliant_Healers_New_Life_in_the_Shadows": "https://www.anime-planet.com/anime/the-brilliant-healers-new-life-in-the-shadows/characters",
    "The_Shiunji_Family_Children": "https://www.anime-planet.com/anime/the-shiunji-family-children/characters",
    "The_Unaware_Atelier_Meister": "https://www.anime-planet.com/anime/the-unaware-atelier-meister/characters",
    "Wind_Breaker": "https://www.anime-planet.com/anime/wind-breaker/characters",
    "Zatsutabi_Thats_Journey": "https://www.anime-planet.com/anime/zatsutabi-thats-journey/characters",
}

ANIME_ALIASES: Dict[str, str] = {
    "Frieren_Beyond_Journey_s_End": "Frieren_Beyond_Journeys_End",
    "Frieren_Beyond_Journey_End": "Frieren_Beyond_Journeys_End",
    "Rascal_Does_Not_Dream": "Rascal_Does_Not_Dream_of_Bunny_Girl_Senpai",
    "Windbreaker": "Wind_Breaker",
    "Wind_Breakers": "Wind_Breaker",
}

STRICT_MAPPING = True

STANDARD_CHAR_SIZE = 384  # All character images resized to this square size

# Load character descriptions
CHARACTER_DESCRIPTIONS = {}
try:
    with open('character_descriptions.json', 'r') as f:
        CHARACTER_DESCRIPTIONS = json.load(f)
        for series_key, characters in CHARACTER_DESCRIPTIONS.items():
            for char_key, char_data in characters.items():
                if 'description' in char_data:
                    char_data['description'] = char_data['description'].rstrip('.')
        LOG.info(f"Loaded character descriptions for {len(CHARACTER_DESCRIPTIONS)} series")
except FileNotFoundError:
    LOG.warning("character_descriptions.json not found. Character tagging will be skipped.")
except json.JSONDecodeError as e:
    LOG.error(f"Error parsing character_descriptions.json: {e}")

# ----------------------- Helpers -------------------------------------

def _norm_key(s: str) -> str:
    """Normalize strings for robust substring matching."""
    s = s or ""
    s = s.lower()
    s = s.replace("snippets/", "")
    s = re.sub(r"\.(mp4|mkv|mov|webm|avi)$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _resolve_anime_key_from_video(video_name: str) -> str:
    """Return the ANIME_MAPPING key for this video."""
    vn_norm = _norm_key(video_name)

    for key in ANIME_MAPPING:
        if _norm_key(key) in vn_norm:
            return key

    for alias, main_key in ANIME_ALIASES.items():
        if _norm_key(alias) in vn_norm and main_key in ANIME_MAPPING:
            return main_key

    vn_tokens = set(vn_norm.split("_"))
    best_key, best_overlap = None, 0
    for key in ANIME_MAPPING:
        k_tokens = set(_norm_key(key).split("_"))
        overlap = len(vn_tokens & k_tokens)
        if overlap > best_overlap:
            best_overlap, best_key = overlap, key
    if best_overlap >= 2:
        return best_key

    msg = f"No anime mapping found for video filename '{video_name}' (norm='{vn_norm}')"
    if STRICT_MAPPING:
        LOG.error(msg)
        raise KeyError(msg)
    else:
        LOG.warning(msg)
        return ""

def download_character_image(url: str, output_path: Path) -> bool:
    """Download character image from URL."""
    try:
        response = requests.get(url, stream=True, timeout=20)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        LOG.warning(f"Failed to download character image {url}: {e}")
        return False

def square_crop_from_top(img: np.ndarray) -> np.ndarray:
    """Square crop image from top, keeping full width and cropping bottom."""
    h, w = img.shape[:2]
    if h <= w:
        return img
    
    crop_size = w
    return img[:crop_size, :]

def resize_to_standard_size(img: np.ndarray, size: int = STANDARD_CHAR_SIZE) -> np.ndarray:
    """Resize image to standard square size."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)

def stitch_character_grid(images: List[np.ndarray]) -> Optional[np.ndarray]:
    """Stitch character images in appropriate grid layout based on count."""
    if not images:
        return None
    
    count = len(images)
    
    if count == 1:
        return images[0]
    
    elif count == 2:
        # 1x2 horizontal
        return np.hstack(images)
    
    elif count == 3:
        # 1x3 horizontal
        return np.hstack(images)
    
    elif count == 4:
        # 2x2 grid
        top_row = np.hstack(images[:2])
        bottom_row = np.hstack(images[2:4])
        return np.vstack([top_row, bottom_row])
    
    elif count == 5:
        # 3+2 layout (3 on top, 2 centered on bottom)
        top_row = np.hstack(images[:3])
        bottom_row = np.hstack(images[3:5])
        
        # Center bottom row
        top_width = top_row.shape[1]
        bottom_width = bottom_row.shape[1]
        
        if bottom_width < top_width:
            padding = (top_width - bottom_width) // 2
            bottom_row = cv2.copyMakeBorder(
                bottom_row,
                0, 0, padding, top_width - bottom_width - padding,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            )
        
        return np.vstack([top_row, bottom_row])
    
    elif count == 6:
        # 3x2 grid
        top_row = np.hstack(images[:3])
        bottom_row = np.hstack(images[3:6])
        return np.vstack([top_row, bottom_row])
    
    else:
        # Fallback: horizontal stitch
        LOG.warning(f"Unexpected character count {count}, using horizontal stitch")
        return np.hstack(images)

def tag_characters_in_prompt(description: str, anime_key: str) -> Tuple[str, List[str]]:
    """Tag character names in description with their descriptions."""
    if not CHARACTER_DESCRIPTIONS or anime_key not in CHARACTER_DESCRIPTIONS:
        return description, []
    
    anime_characters = CHARACTER_DESCRIPTIONS[anime_key]
    tagged_description = description
    found_characters = []
    
    SKIP_NAMES = {'the', 'a', 'an', 'person', 'man', 'woman', 'girl', 'boy', 'guide'}
    
    sorted_chars = sorted(anime_characters.items(), 
                         key=lambda x: len(x[1]['first_name']), 
                         reverse=True)
    
    for char_key, char_data in sorted_chars:
        first_name = char_data['first_name']
        
        if first_name.lower() in SKIP_NAMES:
            continue
            
        pattern = re.compile(r'\b' + re.escape(first_name) + r'\b', re.IGNORECASE)
        
        if pattern.search(tagged_description):
            def replacer(match):
                return f"{match.group()} ({char_data['description']})"
            
            tagged_description = pattern.sub(replacer, tagged_description, count=1)
            found_characters.append(char_key)
    
    return tagged_description, found_characters

def parse_ndjson(filepath: str, accepted_statuses: List[str] = ['DONE']):
    """
    Parse NDJSON file and return list of annotated videos.
    Only includes videos with accepted workflow status.
    """
    annotated_videos = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)

                for project_id, project_data in data.get('projects', {}).items():
                    if project_data.get('labels'):
                        workflow_status = project_data.get('project_details', {}).get('workflow_status', 'UNKNOWN')
                        
                        if workflow_status in accepted_statuses:
                            annotated_videos.append({
                                'video_url': data['data_row']['row_data'],
                                'video_name': data['data_row']['global_key'],
                                'labels': project_data['labels'],
                                'media_attributes': data.get('media_attributes', {}),
                                'workflow_status': workflow_status
                            })

    return annotated_videos

def download_video(video_url: str, sas_token: str, output_path: str) -> str:
    """Download video from Azure Blob Storage with SAS token."""
    video_url = video_url.replace('/snippets//', '/snippets/')

    if '?' in video_url:
        full_url = f"{video_url}&{sas_token}"
    else:
        full_url = f"{video_url}?{sas_token}"

    LOG.info(f"Downloading video: {full_url[:120]}...")
    response = requests.get(full_url, stream=True, timeout=60)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return output_path

def ffprobe_fps(video_path: str) -> float:
    """Get FPS via ffprobe as fallback."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate",
             "-of", "default=nk=1:nw=1", video_path],
            text=True
        ).strip()
        if "/" in out:
            a, b = out.split("/")
            b = float(b)
            return float(a) / b if b else 0.0
        return float(out) if out else 0.0
    except Exception:
        return 0.0

def extract_frame_ffmpeg(video_path: str, timestamp_sec: float, output_path: Path) -> bool:
    """Extract a specific frame at timestamp using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{timestamp_sec:.6f}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        "-y", str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True)
        return output_path.exists() and output_path.stat().st_size > 0
    except subprocess.CalledProcessError:
        return False

def format_text_annotation(frame_data, frame_number, current_scene):
    """Format the annotation text according to the template."""
    classifications = frame_data.get('classifications', [])

    scene_change = "Same scene"
    location = current_scene.get('location', '')
    inside_outside = current_scene.get('inside_outside', '')
    time_of_day = current_scene.get('time_of_day', '')
    text_description = ""
    shot_size = ""
    shot_framing = ""
    shot_angle = ""

    for classification in classifications:
        feature_name = classification.get('name', '')

        if feature_name == 'Shot Description':
            text_description = classification.get('text_answer', {}).get('content', '')

            for nested in classification.get('text_answer', {}).get('classifications', []):
                nested_name = nested.get('name', '')

                if nested_name == 'Shot Size':
                    shot_size_value = nested.get('radio_answer', {}).get('value', '')
                    shot_size = shot_size_value.replace('_', ' ').title()

                elif nested_name == 'Shot Framing':
                    shot_framing_value = nested.get('radio_answer', {}).get('value', '')
                    shot_framing = shot_framing_value.replace('_', ' ').title()

                elif nested_name == 'Shot Angle':
                    shot_angle_value = nested.get('radio_answer', {}).get('value', '')
                    shot_angle = shot_angle_value.replace('_', ' ').title()

                elif nested_name == 'Scene Change':
                    scene_value = nested.get('radio_answer', {}).get('value', '')
                    if scene_value == 'new_scene':
                        scene_change = "New scene"

                        for scene_nested in nested.get('radio_answer', {}).get('classifications', []):
                            if scene_nested.get('name') == 'Location':
                                location = scene_nested.get('text_answer', {}).get('content', '')

                                for loc_nested in scene_nested.get('text_answer', {}).get('classifications', []):
                                    if loc_nested.get('name') == 'Inside or Outside?':
                                        inside_outside = loc_nested.get('radio_answer', {}).get('value', '').capitalize()
                                    elif loc_nested.get('name') == 'Time of Day':
                                        time_of_day = loc_nested.get('radio_answer', {}).get('value', '').capitalize()

                        current_scene['location'] = location
                        current_scene['inside_outside'] = inside_outside
                        current_scene['time_of_day'] = time_of_day

    lines = []

    shot_details = []
    if text_description:
        shot_details.append(text_description)
    if shot_size:
        shot_details.append(shot_size)
    if shot_framing:
        shot_details.append(shot_framing)
    if shot_angle:
        shot_details.append(shot_angle)
    
    if shot_details:
        lines.append(", ".join(shot_details))
    
    scene_line = scene_change
    if location:
        scene_line += f": {location}"
        details = []
        if inside_outside:
            details.append(inside_outside)
        if time_of_day:
            details.append(time_of_day)
        if details:
            scene_line += f" ({', '.join(details)})"
    lines.append(scene_line)

    return "\n".join(lines), current_scene

def clean_video_name(video_name: str) -> str:
    """Clean video name for use as folder name."""
    name = video_name.replace('snippets/', '')
    if name.lower().endswith('.mp4'):
        name = name[:-4]
    name = name.replace(' ', '_').replace('/', '_')
    return name

def pick_segment_starts(label: dict) -> List[int]:
    """Extract first frame of each labeled segment."""
    frames = []
    annotations = label.get('annotations', {})
    segments = annotations.get('segments', {})
    
    for feature_name, ranges in segments.items():
        for r in ranges:
            if isinstance(r, (list, tuple)) and r:
                frames.append(int(r[0]))  # First frame of segment
    
    return sorted(set(frames))

def create_training_samples(video_frames, anime_key: str, video_name: str, output_dir: Path, global_sample_num: int) -> Tuple[int, List[Path], List[dict], List[dict]]:
    """Create training samples with character images and separate context frames."""
    sample_num = global_sample_num
    metadata_entries = []
    series_tracking = []
    
    anime_characters = CHARACTER_DESCRIPTIONS.get(anime_key.lower().replace("-", "_"), {})
    character_image_cache = {}

    for i in range(len(video_frames)):
        sample_dir = output_dir / f"sample_{sample_num:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Get up to 3 context frames
        start_idx = max(0, i - 3)
        context_frames = video_frames[start_idx:i]
        context_frames = context_frames[-3:]  # Limit to max 3
        
        out_frame_path, text_content = video_frames[i]
        lines = text_content.split('\n')
        description_line = lines[0] if len(lines) > 0 else ""
        scene_line = lines[1] if len(lines) > 1 else ""
        
        # Find mentioned characters
        mentioned_characters = []
        if anime_key.lower().replace("-", "_") in CHARACTER_DESCRIPTIONS:
            anime_chars = CHARACTER_DESCRIPTIONS[anime_key.lower().replace("-", "_")]
            sorted_chars = sorted(anime_chars.items(), 
                                 key=lambda x: len(x[1]['first_name']), 
                                 reverse=True)
            
            SKIP_NAMES = {'the', 'a', 'an', 'person', 'man', 'woman', 'girl', 'boy', 'guide'}
            
            for char_key, char_data in sorted_chars:
                first_name = char_data['first_name']
                if first_name.lower() not in SKIP_NAMES:
                    pattern = re.compile(r'\b' + re.escape(first_name) + r'\b', re.IGNORECASE)
                    if pattern.search(description_line):
                        mentioned_characters.append(char_key)
        
        # Process character images (no borders)
        processed_char_images = []
        
        for char_key in mentioned_characters:
            if char_key in anime_characters:
                char_data = anime_characters[char_key]
                if char_data.get('image_url'):
                    # Download if not cached
                    if char_key not in character_image_cache:
                        temp_char_path = Path(tempfile.mktemp(suffix='_char.jpg'))
                        if download_character_image(char_data['image_url'], temp_char_path):
                            character_image_cache[char_key] = temp_char_path
                            LOG.debug(f"Downloaded and cached character {char_data['first_name']}")
                    
                    if char_key in character_image_cache:
                        # Load, square crop, and resize to standard size (no border)
                        img = cv2.imread(str(character_image_cache[char_key]))
                        if img is not None:
                            img = square_crop_from_top(img)
                            img = resize_to_standard_size(img)
                            processed_char_images.append(img)
        
        # Tag characters (without color information)
        tagged_description, _ = tag_characters_in_prompt(description_line, anime_key.lower().replace("-", "_"))
        
        # Save character image if we have any
        edit_images = []
        image_num = 1
        
        if processed_char_images:
            char_grid = stitch_character_grid(processed_char_images)
            if char_grid is not None:
                char_path = sample_dir / f"{image_num}.jpg"
                cv2.imwrite(str(char_path), char_grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
                edit_images.append(f"sample_{sample_num:03d}/{image_num}.jpg")
                image_num += 1
                LOG.debug(f"Created character grid with {len(processed_char_images)} characters")
        
        # Save context frames as separate images
        for frame_path, _ in context_frames:
            img = cv2.imread(str(frame_path))
            if img is not None:
                context_path = sample_dir / f"{image_num}.jpg"
                cv2.imwrite(str(context_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                edit_images.append(f"sample_{sample_num:03d}/{image_num}.jpg")
                image_num += 1
        
        # Save output frame
        img = cv2.imread(str(out_frame_path))
        if img is not None:
            cv2.imwrite(str(sample_dir / "out.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Create prompt
        prompt_text = f"Create the next shot in the sequence: {tagged_description}. {scene_line}"
        
        metadata_entry = {
            "image": f"sample_{sample_num:03d}/out.jpg",
            "prompt": prompt_text,
            "edit_image": edit_images
        }
        metadata_entries.append(metadata_entry)
        
        # Track series for this sample
        series_tracking.append({
            "sample": f"sample_{sample_num:03d}",
            "series": anime_key,
            "video": video_name
        })

        LOG.info(f"Created sample_{sample_num:03d}: {len(context_frames)} ctx frames, "
                f"{len(processed_char_images)} chars → {len(edit_images)} input images")
        sample_num += 1
    
    return sample_num, list(character_image_cache.values()), metadata_entries, series_tracking

def process_video_for_training(video_data, sas_token: str, output_dir: Path, global_sample_num: int) -> Tuple[int, List[dict], List[dict]]:
    """Process a single video for training dataset creation."""
    video_name = clean_video_name(video_data['video_name'])
    LOG.info(f"Processing video: '{video_name}'")

    try:
        anime_key = _resolve_anime_key_from_video(video_name)
        LOG.info(f"Mapped to anime: {anime_key}")
    except KeyError as e:
        if STRICT_MAPPING:
            raise
        anime_key = ""

    video_metadata = []
    video_series_tracking = []

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        try:
            video_path = download_video(video_data['video_url'], sas_token, tmp_file.name)

            # Get FPS from export
            fps = video_data.get('media_attributes', {}).get('frame_rate')
            try:
                fps = float(fps) if fps is not None else 0.0
            except:
                fps = 0.0
            
            if not fps or fps <= 0:
                fps = ffprobe_fps(video_path)
            
            if not fps or fps <= 0:
                LOG.error(f"FPS unavailable for {video_name}; skipping")
                return global_sample_num, video_metadata
            
            LOG.info(f"Using FPS: {fps}")

            for label in video_data['labels']:
                # Get frames from segments (first frame of each range)
                frame_indices = pick_segment_starts(label)
                
                if not frame_indices:
                    LOG.warning(f"No segments found for {video_name}")
                    continue
                
                LOG.info(f"Found {len(frame_indices)} segment starts to process")

                current_scene = {
                    'location': '',
                    'inside_outside': '',
                    'time_of_day': ''
                }

                video_frames = []
                temp_frame_dir = Path(tempfile.mkdtemp())

                # Get frame data for annotation parsing
                annotations = label.get('annotations', {})
                frames_data = annotations.get('frames', {})

                for idx, frame_num in enumerate(frame_indices):
                    # Convert frame index to timestamp (Labelbox uses 1-based frame numbering)
                    timestamp = (frame_num - 1) / fps
                    
                    frame_path = temp_frame_dir / f"frame_{idx:03d}.png"
                    success = extract_frame_ffmpeg(video_path, timestamp, frame_path)

                    if success:
                        # Get annotation data for this frame
                        frame_data = frames_data.get(str(frame_num), {})
                        text_content, current_scene = format_text_annotation(frame_data, frame_num, current_scene)
                        video_frames.append((frame_path, text_content))
                        LOG.debug(f"Extracted frame {frame_num} at t={timestamp:.3f}s")
                    else:
                        LOG.warning(f"Failed to extract frame {frame_num} from '{video_name}'")

                if video_frames:
                    global_sample_num, temp_char_images, metadata_entries, series_tracking = create_training_samples(
                        video_frames, anime_key, video_name, output_dir, global_sample_num
                    )
                    video_metadata.extend(metadata_entries)
                    video_series_tracking.extend(series_tracking)
                    
                    for char_img in temp_char_images:
                        try:
                            if char_img.exists():
                                char_img.unlink()
                        except:
                            pass

                import shutil
                shutil.rmtree(temp_frame_dir)

        finally:
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)

    return global_sample_num, video_metadata, video_series_tracking

# ------------------------------ Main ---------------------------------

def main():
    NDJSON_FILE = os.environ.get("LABELBOX_NDJSON_FILE", "labelbox_export_20251018_225034.ndjson")
    SAS_TOKEN = os.environ.get("AZURE_SAS_TOKEN")
    OUTPUT_DIR = Path("dataset")
    
    ACCEPTED_STATUSES = ['DONE']

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOG.info(f"Parsing NDJSON file (accepting workflow_status: {ACCEPTED_STATUSES})...")
    annotated_videos = parse_ndjson(NDJSON_FILE, accepted_statuses=ACCEPTED_STATUSES)
    LOG.info(f"Found {len(annotated_videos)} annotated videos with accepted workflow status")
    
    if annotated_videos:
        status_counts = {}
        for video in annotated_videos:
            status = video.get('workflow_status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
        LOG.info(f"Workflow status distribution: {status_counts}")

    global_sample_num = 1
    all_metadata = []
    all_series_tracking = []

    for i, video_data in enumerate(annotated_videos, 1):
        video_name = clean_video_name(video_data['video_name'])
        LOG.info(f"[{i}/{len(annotated_videos)}] Processing: {video_name} (status: {video_data.get('workflow_status', 'UNKNOWN')})")

        try:
            global_sample_num, video_metadata, video_series_tracking = process_video_for_training(video_data, SAS_TOKEN, OUTPUT_DIR, global_sample_num)
            all_metadata.extend(video_metadata)
            all_series_tracking.extend(video_series_tracking)
            LOG.info(f"✓ Completed {video_name}")
        except KeyError as e:
            LOG.error(f"✗ Mapping error for {video_name}: {e}")
            continue
        except Exception as e:
            LOG.exception(f"✗ Error processing {video_name}: {e}")
            continue

    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    # Write series tracking CSV
    series_tracking_path = Path("series_tracking.csv")
    with open(series_tracking_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample', 'series', 'video'])
        writer.writeheader()
        writer.writerows(all_series_tracking)
    
    LOG.info("Training dataset creation complete!")
    LOG.info(f"Total samples created: {global_sample_num - 1}")
    LOG.info(f"Metadata written to: {metadata_path}")
    LOG.info(f"Series tracking written to: {series_tracking_path}")

if __name__ == "__main__":
    main()