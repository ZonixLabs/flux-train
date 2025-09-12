#!/usr/bin/env python3
"""
One-time script to generate character descriptions using GPT-4o vision
Creates character_descriptions.json
"""

import json
import os
import time
import base64
import requests
from pathlib import Path
from typing import Dict, Optional
from openai import OpenAI
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("character_descriptor")

# Anime mapping copied from main script
ANIME_MAPPING = {
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

# Try to use curl_cffi if available
try:
    from curl_cffi import requests as cffi_requests
except ImportError:
    LOG.warning("curl_cffi not installed. May fail on Cloudflare-protected sites.")
    cffi_requests = None

PROMPT = """Analyze this anime character image and provide a brief natural description.

For human or humanoid characters, describe their gender and hair:
- Gender: man, woman, or person (if ambiguous)
- Hair: color, include "long" if it reaches shoulders or beyond, or "bald"
- Age: only mention if obviously elderly or a young child

For non-humanoid entities (creatures, robots, etc.), provide a simple descriptive phrase.

Examples:
- the woman with long purple hair
- the man with red hair
- the bald man
- the elderly woman with white hair
- the young boy with black hair
- the blue mechanical dragon
- the small cat creature

Provide only the description in this natural format and do not add anything more."""

def fetch_character_page(url: str) -> Optional[BeautifulSoup]:
    """Fetch and parse an anime character page."""
    try:
        if cffi_requests:
            response = cffi_requests.get(url, impersonate="chrome")
        else:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except Exception as e:
        LOG.error(f"Error fetching {url}: {e}")
        return None

def extract_characters(soup: BeautifulSoup) -> Dict[str, Dict[str, Optional[str]]]:
    """Extract character names and images from anime-planet page."""
    characters = {}

    # Primary structure: tables under <h3 class="sub">
    categories = soup.find_all('h3', class_='sub')
    for category in categories:
        table = category.find_next_sibling('table')
        if not table:
            continue

        for row in table.find_all('tr'):
            # Get character image
            img_cell = row.find('td', class_='tableAvatar')
            img_url = None
            if img_cell:
                img = img_cell.find('img')
                if img and img.get('src'):
                    img_url = img['src']
                    if img_url and not img_url.startswith('http'):
                        img_url = urljoin('https://www.anime-planet.com', img_url)

            # Get character name
            name_cell = row.find('td', class_='tableCharInfo')
            if name_cell:
                name_link = name_cell.find('a', class_='name')
                if name_link:
                    full_name = name_link.get_text(strip=True)
                    if full_name:
                        first_name = full_name.split()[0]
                        key = first_name.lower()
                        if key not in characters:
                            characters[key] = {'name': full_name, 'image': img_url}

    # Fallback structure
    if not characters:
        for a in soup.select("a[href*='/character/']"):
            full_name = a.get_text(strip=True)
            if not full_name:
                continue
            img = a.find('img')
            img_url = img.get('src') if img and img.get('src') else None
            if img_url and not img_url.startswith('http'):
                img_url = urljoin('https://www.anime-planet.com', img_url)
            key = full_name.split()[0].lower()
            if key not in characters:
                characters[key] = {'name': full_name, 'image': img_url}

    return characters

def download_image_to_base64(url: str) -> Optional[str]:
    """Download image and convert to base64."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        LOG.error(f"Failed to download {url}: {e}")
        return None

def get_character_description(client: OpenAI, image_base64: str) -> str:
    """Get character description from GPT-4o."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4o-latest" 
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }}
                ]
            }],
            max_tokens=50,
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        LOG.error(f"GPT-4o API error: {e}")
        return ""

def main():
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    client = OpenAI(api_key=api_key)
    
    # Load existing progress if any (for resuming)
    output_file = Path("character_descriptions.json")
    if output_file.exists():
        with open(output_file, 'r') as f:
            all_descriptions = json.load(f)
        LOG.info(f"Resuming from existing file with {len(all_descriptions)} series")
    else:
        all_descriptions = {}
    
    total_characters = 0
    
    # Process each anime series
    for anime_key, url in ANIME_MAPPING.items():
        # Clean up the key for storage (lowercase, underscores)
        storage_key = anime_key.lower().replace("-", "_")
        
        if storage_key in all_descriptions:
            LOG.info(f"Skipping {anime_key} (already processed)")
            total_characters += len(all_descriptions[storage_key])
            continue
            
        LOG.info(f"Processing {anime_key}")
        soup = fetch_character_page(url)
        if not soup:
            LOG.warning(f"Failed to fetch {anime_key}")
            continue
            
        characters = extract_characters(soup)
        LOG.info(f"  Found {len(characters)} characters")
        anime_descriptions = {}
        
        for char_name, char_data in characters.items():
            if not char_data['image']:
                LOG.warning(f"  No image for {char_name}")
                continue
                
            # Download and process image
            image_base64 = download_image_to_base64(char_data['image'])
            if not image_base64:
                continue
                
            # Get description from GPT-4o
            description = get_character_description(client, image_base64)
            if not description:
                continue
                
            first_name = char_data['name'].split()[0]
            anime_descriptions[first_name.lower()] = {
                "full_name": char_data['name'],
                "first_name": first_name,
                "description": description,
                "image_url": char_data['image']
            }
            
            LOG.info(f"  {first_name}: {description}")
            total_characters += 1
            time.sleep(0.5)  # Rate limiting
        
        all_descriptions[storage_key] = anime_descriptions
        
        # Save progress after each series
        with open(output_file, 'w') as f:
            json.dump(all_descriptions, f, indent=2)
        LOG.info(f"Saved progress: {len(all_descriptions)} series, {total_characters} total characters")
    
    LOG.info(f"Complete! Generated descriptions for {total_characters} characters across {len(all_descriptions)} series")
    
    # Estimate cost (rough)
    cost_per_image = 0.00425  # GPT-4o vision pricing estimate
    LOG.info(f"Estimated API cost: ${total_characters * cost_per_image:.2f}")

if __name__ == "__main__":
    main()