import json
import requests
import time
import re
import os
from enum import Enum
from datasets import load_dataset

save_path = "character_classification.json"

en_yue_sentences = load_dataset("google/smol", "smolsent__en_yue")['train'] # There is only a train split here

class CharacterType(Enum):
    UNKNOWN = 0 # Not described in the online dictionary
    CANTO_ONLY = 1 # It is described in the online dictionary as being exclusively Cantonese
    NOT_CANTO_ONLY = 2 # It is described in the online dictionary but is not exclusively Cantonese
    

def classify_character(character: str) -> CharacterType:
    '''Classify the parameter character as CANTO_ONLY, NOT_CANTO_ONLY, or UNKNOWN by querying an online Canto dict that has this description'''
    url = "https://www.cantonese.sheik.co.uk/scripts/wordsearch.php?level=0"

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:144.0) Gecko/20100101 Firefox/144.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "null",
        "Upgrade-Insecure-Requests": "1",
        "Connection": "keep-alive",
    }

    data = {
        "TEXT": character,
        "SEARCHTYPE": "2",
        "radicaldropdown": "0",
        "searchsubmit": "search"
    }

    # Retry logic for first request
    for attempt in range(5):
        try:
            response = requests.post(url, headers=headers, data=data, timeout=10)
            break
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    else:
        return CharacterType.UNKNOWN

    # Extract redirect URL
    match = re.search(r'url=([^"]+)', response.text)
    if not match:
        return CharacterType.UNKNOWN
    redirect_url = match.group(1)

    time.sleep(0.1)

    for attempt in range(5):
        try:
            response = requests.get(redirect_url, headers=headers, timeout=10)
            break
        except requests.exceptions.RequestException as e:
            print(f"Redirect attempt {attempt + 1} failed: {e}")
            time.sleep(3)
    else:
        # The character is not described in the dictionary
        return CharacterType.UNKNOWN 

    # If ths page contains this then it's CANTO_ONLY, otherwise we cannot decide if it is
    if "This character is used in Cantonese, not Mandarin/Standard written Chinese." in response.text:
        return CharacterType.CANTO_ONLY
    else:
        return CharacterType.NOT_CANTO_ONLY


def get_all_chars_in_sentences(sentences) -> set[str]:
    all_chars = set()
    for entry in sentences:
        all_chars.update(set(entry['trg']))
    return all_chars

# This will contain all unique characters in the Cantonese SmolSent sentences
all_chars = list(get_all_chars_in_sentences(en_yue_sentences))

def fetch_data_for_chars(chars: list[str]) -> dict[str, CharacterType]:
    '''Fetch character type data for a list of characters'''
    dict = {}
    for i, char in enumerate(all_chars):
        print(f"Processing character {i+1}/{len(all_chars)}: {char}")
        dict[char] = classify_character(char)
    return dict

def load_character_type_data() -> dict[str, CharacterType]:
    '''Load character type data from file or fetch it if not available'''
    with open(save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {k: CharacterType[v] for k, v in data.items()}

def save_character_type_data(data: dict[str, CharacterType]) -> None:
    '''Save character type data to file'''
    serializable = {k: v.name for k, v in data.items()}
    with open(save_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(serializable, f, ensure_ascii=False, indent=4)
        
def print_summary(character_type_dict: dict[str, CharacterType]):
    '''Print a summary of the character type data.'''
    canto_only = 0
    not_canto_only = 0
    unknown = 0
    for char, char_type in character_type_dict.items():
        if char_type == CharacterType.CANTO_ONLY:
            canto_only += 1
        elif char_type == CharacterType.NOT_CANTO_ONLY:
            not_canto_only += 1
        else:
            unknown += 1
            
    print(f"Cantonese-only characters: {canto_only}")
    print(f"Not Cantonese-only characters: {not_canto_only}")
    print(f"Unknown characters: {unknown}")

    # all canto-only characters
    canto_only_chars = set([char for char, char_type in character_type_dict.items() if char_type == CharacterType.CANTO_ONLY])
    print("Cantonese-only characters:", canto_only_chars)
    
    sentences_with_canto_only = 0
    for entry in en_yue_sentences:
        if any(char in canto_only_chars for char in entry['trg']): # type: ignore
            sentences_with_canto_only += 1
        
    print(f"Sentences with at least one Cantonese-only character: {sentences_with_canto_only} out of {len(en_yue_sentences)}") # type: ignore

# Load or fetch character type data
if os.path.exists(save_path):
    character_type_dict = load_character_type_data()
else:
    character_type_dict = fetch_data_for_chars(all_chars)
    save_character_type_data(character_type_dict)
    
print_summary(character_type_dict)