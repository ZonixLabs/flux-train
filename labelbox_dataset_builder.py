#!/usr/bin/env python3
"""
Script to parse Labelbox NDJSON annotations and create a training dataset with character integration.

- Character descriptions loaded from character_descriptions.json
- Characters integrated into context frames instead of separate assets
- Updated prompt format with "Create the next shot:" prefix

Requirements:
    pip install beautifulsoup4 requests opencv-python
    # optional (helps with Cloudflare)
    pip install curl-cffi
"""

import json
import os
import logging
import requests
from pathlib import Path
import cv2
import tempfile
from urllib.parse import urljoin
import re
import time
from typing import Dict, List, Tuple, Optional, Set
from bs4 import BeautifulSoup

# --------------------------- Logging ---------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOG = logging.getLogger("dataset_builder")

# ----------------------- Optional curl_cffi ---------------------------

try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    LOG.warning("curl_cffi not installed. Character scraping might fail on Cloudflare-protected sites. "
                "Install with: pip install curl-cffi")
    cffi_requests = None

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
    "Summer_Pockets": "https://www.anime-planet.com/anime/summer-pockets/characters",
    "The_Apothecary_Diaries": "https://www.anime-planet.com/anime/the-apothecary-diaries/characters",
    "The_Brilliant_Healers_New_Life_in_the_Shadows": "https://www.anime-planet.com/anime/the-brilliant-healers-new-life-in-the-shadows/characters",
    "The_Shiunji_Family_Children": "https://www.anime-planet.com/anime/the-shiunji-family-children/characters",
    "The_Unaware_Atelier_Meister": "https://www.anime-planet.com/anime/the-unaware-atelier-meister/characters",
    "Wind_Breaker": "https://www.anime-planet.com/anime/wind-breaker/characters",
    "Zatsutabi_Thats_Journey": "https://www.anime-planet.com/anime/zatsutabi-thats-journey/characters",
}

# Optional alias map (helps with common alt spellings). Keys = alias, values = primary key in ANIME_MAPPING
ANIME_ALIASES: Dict[str, str] = {
    # Frieren apostrophe variants
    "Frieren_Beyond_Journey_s_End": "Frieren_Beyond_Journeys_End",
    "Frieren_Beyond_Journey_End": "Frieren_Beyond_Journeys_End",
    # Bunny Girl Senpai shorthand
    "Rascal_Does_Not_Dream": "Rascal_Does_Not_Dream_of_Bunny_Girl_Senpai",
    # Wind Breaker variants
    "Windbreaker": "Wind_Breaker",
    "Wind_Breakers": "Wind_Breaker",
}

# If True, a video without a mapping raises an error (and is logged). If False, it only logs and skips.
STRICT_MAPPING = True

# Load character descriptions
CHARACTER_DESCRIPTIONS = {}
try:
    with open('character_descriptions.json', 'r') as f:
        CHARACTER_DESCRIPTIONS = json.load(f)
        # Strip trailing periods from all descriptions when loading
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
    """
    Normalize strings for robust substring matching:
    - lowercase
    - remove leading 'snippets/'
    - strip common video extensions
    - replace any non-alphanumeric with single underscores
    """
    s = s or ""
    s = s.lower()
    s = s.replace("snippets/", "")
    s = re.sub(r"\.(mp4|mkv|mov|webm|avi)$", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _resolve_anime_key_from_video(video_name: str) -> str:
    """
    Return the ANIME_MAPPING key for this video.
    Raises a KeyError if no mapping is found and STRICT_MAPPING=True.
    """
    vn_norm = _norm_key(video_name)

    # 1) Exact normalized substring match against mapping keys
    for key in ANIME_MAPPING:
        if _norm_key(key) in vn_norm:
            return key

    # 2) Alias matching
    for alias, main_key in ANIME_ALIASES.items():
        if _norm_key(alias) in vn_norm and main_key in ANIME_MAPPING:
            return main_key

    # 3) Token-overlap fallback
    vn_tokens = set(vn_norm.split("_"))
    best_key, best_overlap = None, 0
    for key in ANIME_MAPPING:
        k_tokens = set(_norm_key(key).split("_"))
        overlap = len(vn_tokens & k_tokens)
        if overlap > best_overlap:
            best_overlap, best_key = overlap, key
    # Require minimum overlap (>=2 tokens) to avoid false positives
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

def extract_characters_from_text(text: str, anime_characters: Dict) -> Set[str]:
    """Extract character names mentioned in text."""
    found_characters = set()
    text_lower = text.lower()
    
    for char_key, char_data in anime_characters.items():
        # Check for character's first name in text
        if char_key in text_lower or char_data['first_name'].lower() in text_lower:
            found_characters.add(char_key)
    
    return found_characters

def tag_characters_in_prompt(description: str, anime_key: str) -> Tuple[str, List[str]]:
    """
    Tag character names in description with their descriptions.
    Returns (tagged_description, list_of_character_keys_found)
    """
    if not CHARACTER_DESCRIPTIONS or anime_key not in CHARACTER_DESCRIPTIONS:
        return description, []
    
    anime_characters = CHARACTER_DESCRIPTIONS[anime_key]
    tagged_description = description
    found_characters = []
    
    # Sort characters by name length (longest first) to avoid partial replacements
    sorted_chars = sorted(anime_characters.items(), 
                         key=lambda x: len(x[1]['first_name']), 
                         reverse=True)
    
    for char_key, char_data in sorted_chars:
        first_name = char_data['first_name']
        # Case-insensitive search but preserve original case in replacement
        pattern = re.compile(r'\b' + re.escape(first_name) + r'\b', re.IGNORECASE)
        
        if pattern.search(tagged_description):
            # Replace first occurrence only with description
            def replacer(match):
                return f"{match.group()} ({char_data['description']})"
            
            tagged_description = pattern.sub(replacer, tagged_description, count=1)
            found_characters.append(char_key)
    
    return tagged_description, found_characters

def parse_ndjson(filepath: str, accepted_statuses: List[str] = ['DONE']):
    """
    Parse NDJSON file and return list of annotated videos.
    
    Args:
        filepath: Path to NDJSON file
        accepted_statuses: List of workflow statuses to accept (default: ['DONE'])
    """
    annotated_videos = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)

                # Check if this video has annotations
                for project_id, project_data in data.get('projects', {}).items():
                    if project_data.get('labels'):
                        # Check workflow status at PROJECT level, not label level
                        workflow_status = project_data.get('project_details', {}).get('workflow_status', 'UNKNOWN')
                        
                        # Only include if status is in accepted list
                        if workflow_status in accepted_statuses:
                            annotated_videos.append({
                                'video_url': data['data_row']['row_data'],
                                'video_name': data['data_row']['global_key'],
                                'labels': project_data['labels'],
                                'frame_count': data.get('media_attributes', {}).get('frame_count'),
                                'workflow_status': workflow_status
                            })

    return annotated_videos
    
def download_video(video_url: str, sas_token: str, output_path: str) -> str:
    """Download video from Azure Blob Storage with SAS token."""
    # Fix double slash issue if present
    video_url = video_url.replace('/snippets//', '/snippets/')

    # Append SAS token to URL
    if '?' in video_url:
        full_url = f"{video_url}&{sas_token}"
    else:
        full_url = f"{video_url}?{sas_token}"

    LOG.info(f"Downloading video: {full_url[:120]}...")  # truncate for log
    response = requests.get(full_url, stream=True, timeout=60)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    return output_path

def extract_frame(video_path: str, frame_number: int, output_path: Path) -> bool:
    """Extract a specific frame from video using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if ret and frame is not None:
        cv2.imwrite(str(output_path), frame)
        cap.release()
        return True
    else:
        cap.release()
        return False

def format_text_annotation(frame_data, frame_number, current_scene):
    """Format the annotation text according to the new template."""
    classifications = frame_data.get('classifications', [])

    # Initialize variables
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

            # Parse nested classifications for shot details
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

                        # Get location details
                        for scene_nested in nested.get('radio_answer', {}).get('classifications', []):
                            if scene_nested.get('name') == 'Location':
                                location = scene_nested.get('text_answer', {}).get('content', '')

                                # Get inside/outside and time of day
                                for loc_nested in scene_nested.get('text_answer', {}).get('classifications', []):
                                    if loc_nested.get('name') == 'Inside or Outside?':
                                        inside_outside = loc_nested.get('radio_answer', {}).get('value', '').capitalize()
                                    elif loc_nested.get('name') == 'Time of Day':
                                        time_of_day = loc_nested.get('radio_answer', {}).get('value', '').capitalize()

                        # Update current scene for future frames
                        current_scene['location'] = location
                        current_scene['inside_outside'] = inside_outside
                        current_scene['time_of_day'] = time_of_day

    # Format the output with new structure
    lines = []

    # Build the main description line with shot details
    shot_details = []
    if text_description:
        shot_details.append(text_description)
    if shot_size:
        shot_details.append(shot_size)
    if shot_framing:
        shot_details.append(shot_framing)
    if shot_angle:
        shot_details.append(shot_angle)
    
    # Main description (will be prefixed with "Create the next shot:" later)
    if shot_details:
        lines.append(", ".join(shot_details))
    
    # Scene information goes at the bottom
    scene_line = scene_change
    if location:
        scene_line += f": {location}"
        if inside_outside or time_of_day:
            scene_line += f" - {inside_outside}"
            if time_of_day:
                scene_line += f" - {time_of_day}"
    lines.append(scene_line)

    return "\n".join(lines), current_scene

def clean_video_name(video_name: str) -> str:
    """Clean video name for use as folder name."""
    name = video_name.replace('snippets/', '')
    if name.lower().endswith('.mp4'):
        name = name[:-4]
    # Replace spaces and special characters
    name = name.replace(' ', '_').replace('/', '_')
    return name

def create_training_samples(video_frames, anime_key: str, output_dir: Path, global_sample_num: int) -> Tuple[int, List[Path]]:
    """
    Create training samples with sliding window approach and character integration.
    Returns (next_sample_num, list_of_temp_character_images_to_cleanup)
    """
    sample_num = global_sample_num
    
    # Get anime characters if available
    anime_characters = CHARACTER_DESCRIPTIONS.get(anime_key.lower().replace("-", "_"), {})
    
    # Cache for downloaded character images to avoid re-downloading
    character_image_cache = {}

    # video_frames is list of (frame_path, text_content) tuples
    for i in range(len(video_frames)):
        sample_dir = output_dir / f"sample_{sample_num:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Create in/ directory
        (sample_dir / "in").mkdir(exist_ok=True)

        # Determine context frames (max 4)
        start_idx = max(0, i - 4)
        context_frames = video_frames[start_idx:i]
        
        # Track which characters appear in context frames
        characters_in_context = set()
        for _, context_text in context_frames:
            # Extract just the description part (first line before scene info)
            desc_lines = context_text.split('\n')
            if desc_lines:
                found_chars = extract_characters_from_text(desc_lines[0], anime_characters)
                characters_in_context.update(found_chars)
        
        # Process current frame text
        out_frame_path, text_content = video_frames[i]
        lines = text_content.split('\n')
        description_line = lines[0] if len(lines) > 0 else ""
        scene_line = lines[1] if len(lines) > 1 else ""
        
        # Tag characters in description and get list of characters mentioned
        tagged_description, mentioned_characters = tag_characters_in_prompt(description_line, anime_key.lower().replace("-", "_"))
        
        # Determine which characters need to be added to context
        characters_to_add = []
        for char_key in mentioned_characters:
            if char_key not in characters_in_context and char_key in anime_characters:
                characters_to_add.append(char_key)
        
        # Prepare all images for in/ directory
        in_images = []
        
        # Add character images first (at position 1)
        for char_key in characters_to_add:
            char_data = anime_characters[char_key]
            if char_data.get('image_url'):
                # Check cache first
                if char_key not in character_image_cache:
                    # Download character image to temp location
                    temp_char_path = Path(tempfile.mktemp(suffix='_char.jpg'))
                    if download_character_image(char_data['image_url'], temp_char_path):
                        character_image_cache[char_key] = temp_char_path
                        LOG.debug(f"Downloaded and cached character {char_data['first_name']}")
                
                if char_key in character_image_cache:
                    in_images.append(character_image_cache[char_key])
                    LOG.debug(f"Added character {char_data['first_name']} to context")
        
        # Add context frames after character images
        for frame_path, _ in context_frames:
            in_images.append(frame_path)
        
        # Copy all images to in/ directory with proper numbering
        for j, img_path in enumerate(in_images, 1):
            in_path = sample_dir / "in" / f"{j}.jpg"
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imwrite(str(in_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Copy target frame as out.jpg
        img = cv2.imread(str(out_frame_path))
        if img is not None:
            cv2.imwrite(str(sample_dir / "out.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Create final prompt with "Create the next shot:" prefix
        prompt_text = f"Create the next shot: {tagged_description}"
        if scene_line:
            prompt_text += f"\n{scene_line}"
        
        # Save prompt
        with open(sample_dir / "prompt.txt", 'w', encoding='utf-8') as f:
            f.write(prompt_text)

        LOG.info(f"Created sample_{sample_num:03d}: {len(context_frames)} ctx frames, "
                f"{len(characters_to_add)} char images added, {len(in_images)} total in/")
        sample_num += 1
    
    # Return the character image paths for cleanup later
    return sample_num, list(character_image_cache.values())

def process_video_for_training(video_data, sas_token: str, output_dir: Path, global_sample_num: int) -> int:
    """Process a single video for training dataset creation."""
    video_name = clean_video_name(video_data['video_name'])
    LOG.info(f"Processing video: '{video_name}'")

    # Resolve anime key for character lookup
    try:
        anime_key = _resolve_anime_key_from_video(video_name)
        LOG.info(f"Mapped to anime: {anime_key}")
    except KeyError as e:
        if STRICT_MAPPING:
            raise
        anime_key = ""

    # Download video to temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        try:
            video_path = download_video(video_data['video_url'], sas_token, tmp_file.name)

            # Process each label (should be one per video in this case)
            for label in video_data['labels']:
                annotations = label.get('annotations', {})
                frames = annotations.get('frames', {})

                # Sort frames by frame number
                sorted_frames = sorted(frames.items(), key=lambda x: int(x[0]))

                # Track current scene context across frames
                current_scene = {
                    'location': '',
                    'inside_outside': '',
                    'time_of_day': ''
                }

                # Group consecutive frames and only take the first of each pair/group
                frames_to_process = []
                i = 0
                while i < len(sorted_frames):
                    frame_num, frame_data = sorted_frames[i]
                    frames_to_process.append((frame_num, frame_data))

                    # Skip the next frame if it's consecutive (part of the same annotation range)
                    if i + 1 < len(sorted_frames):
                        next_frame_num = int(sorted_frames[i + 1][0])
                        current_frame_num = int(frame_num)
                        if next_frame_num == current_frame_num + 1:
                            i += 2  # Skip the consecutive frame
                        else:
                            i += 1
                    else:
                        i += 1

                # Extract frames and annotations
                video_frames = []
                temp_frame_dir = Path(tempfile.mkdtemp())

                for idx, (frame_num, frame_data) in enumerate(frames_to_process):
                    frame_number = int(frame_num)

                    # Extract frame
                    frame_path = temp_frame_dir / f"frame_{idx:03d}.png"
                    success = extract_frame(video_path, frame_number, frame_path)

                    if success:
                        # Get text annotation
                        text_content, current_scene = format_text_annotation(frame_data, int(frame_num), current_scene)
                        video_frames.append((frame_path, text_content))
                        LOG.debug(f"Extracted frame {int(frame_num)}")
                    else:
                        LOG.warning(f"Failed to extract frame {int(frame_num)} from '{video_name}'")

                # Create training samples
                if video_frames:
                    global_sample_num, temp_char_images = create_training_samples(video_frames, anime_key, output_dir, global_sample_num)
                    
                    # Clean up temp character images
                    for char_img in temp_char_images:
                        try:
                            if char_img.exists():
                                char_img.unlink()
                        except:
                            pass

                # Cleanup temp frames
                import shutil
                shutil.rmtree(temp_frame_dir)

        finally:
            # Clean up temporary video file
            if os.path.exists(tmp_file.name):
                os.unlink(tmp_file.name)

    return global_sample_num

# ------------------------------ Main ---------------------------------

def main():
    # Configuration
    NDJSON_FILE = os.environ.get("LABELBOX_NDJSON_FILE", "labelbox_export_20250912_185833.ndjson")
    SAS_TOKEN = os.environ.get("AZURE_SAS_TOKEN")
    OUTPUT_DIR = Path("data") / "train"
    
    # Workflow statuses to accept (can be modified as needed)
    # Options: 'DONE', 'IN_REVIEW', 'IN_REWORK', etc.
    ACCEPTED_STATUSES = ['DONE']  # Change this to include other statuses if needed

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse NDJSON file
    LOG.info(f"Parsing NDJSON file (accepting workflow_status: {ACCEPTED_STATUSES})...")
    annotated_videos = parse_ndjson(NDJSON_FILE, accepted_statuses=ACCEPTED_STATUSES)
    LOG.info(f"Found {len(annotated_videos)} annotated videos with accepted workflow status")
    
    if annotated_videos:
        # Log the workflow statuses found
        status_counts = {}
        for video in annotated_videos:
            status = video.get('workflow_status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
        LOG.info(f"Workflow status distribution: {status_counts}")

    # Global sample counter
    global_sample_num = 1

    # Process each video
    for i, video_data in enumerate(annotated_videos, 1):
        video_name = clean_video_name(video_data['video_name'])
        LOG.info(f"[{i}/{len(annotated_videos)}] Processing: {video_name} (status: {video_data.get('workflow_status', 'UNKNOWN')})")

        try:
            global_sample_num = process_video_for_training(video_data, SAS_TOKEN, OUTPUT_DIR, global_sample_num)
            LOG.info(f"✓ Completed {video_name}")
        except KeyError as e:
            # Raised when STRICT_MAPPING=True and we can't resolve an anime mapping
            LOG.error(f"✗ Mapping error for {video_name}: {e}")
            # Continue to next video so one bad filename doesn't stop the batch
            continue
        except Exception as e:
            LOG.exception(f"✗ Error processing {video_name}: {e}")
            continue

    LOG.info("Training dataset creation complete!")
    LOG.info(f"Total samples created: {global_sample_num - 1}")

if __name__ == "__main__":
    main()