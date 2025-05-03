from dotenv import load_dotenv
import os
import requests
import csv
from pathlib import Path

env_path = Path('../.env')
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://www.hybrid-analysis.com/api/v2/feed/detonation?limit=250"

HEADERS = {
    "User-Agent": "Falcon Sandbox",
    "api-key": API_KEY
}

# Function to fetch detonation data
def fetch_detonations():
    print(f"Requesting data from {BASE_URL}")
    try:
        response = requests.get(BASE_URL, headers=HEADERS, timeout=20)
        response.raise_for_status()

        detonations = response.json()

        if not isinstance(detonations, list):
            print("Unexpected response format")
            return []

        print(f"Retrieved {len(detonations)} detonations")
        return detonations

    except requests.exceptions.Timeout:
        print("\U0000274C Request timed out. Try reducing limit or checking network.")
    except requests.exceptions.RequestException as e:
        print(f"\U0000274C Request error: {e}")
    except Exception as e:
        print(f"\U0000274C Other error: {e}")
    return []

# Function to fetch the overview data for a SHA256 hash
def fetch_overview(sha256):
    overview_url = f"https://www.hybrid-analysis.com/api/v2/overview/{sha256}"
    try:
        response = requests.get(overview_url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print(f"\U0000274C Timeout while fetching overview for SHA256 {sha256}")
    except requests.exceptions.RequestException as e:
        print(f"\U0000274C Request error while fetching overview for SHA256 {sha256}: {e}")
    except Exception as e:
        print(f"\U0000274C Other error while fetching overview for SHA256 {sha256}: {e}")
    return {}

# Function to write data to CSV
def write_to_csv(data, filename="detonation_overviews.csv"):
    headers = [
        "sha256", "threat_score", "verdict", "file_size", "type", "vx_family",
        "is_peexe", "is_64bit", "is_executable", "tag_evasive", "scanner_crowdstrike_percent",
        "scanner_metadefender_percent", "scanner_metadefender_positives", "multiscan_result"
    ]
    
    # Open file in write mode, create if it doesn't exist
    with open(filename, mode="w", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        
        # Write header only if the file is empty
        writer.writeheader()

        # Write data rows
        for row in data:
            writer.writerow(row)

# Main function to gather and process data
def process_data():
    samples = fetch_detonations()
    if not samples:
        print("No samples retrieved.")
        return

    overview_data = []

  #  for sample in samples:
   #     sha256 = sample.get('sha256')
    #    if sha256:
        #    overview = fetch_overview(sha256)
         #   if overview:
                # Check if 'scanners_v2' exists and is not None
    for sample in samples:
        sha256 = sample.get('sha256')
        if sha256:
            overview = fetch_overview(sha256)
            if overview:
                # Ensure scanners_v2 exists and isn't None
               # scanners_v2 = overview.get("scanners_v2") or {}

                row = {
                    "sha256": sha256,
                    "threat_score": overview.get('threat_score'),
                    "verdict": overview.get('verdict'),
                    "file_size": overview.get('size'),
                    "type": overview.get('type'),
                    "vx_family": overview.get('vx_family'),
                    #"is_peexe": "PE/EXE" in overview.get('type', ""),  # Basic check for EXE type
                    #"is_64bit": None,  # Assuming this needs to be fetched from another source if available
                    #"is_executable": "executable" in overview.get('type', "").lower(),  # Basic check for executable type
                    #"tag_evasive": "evasive" in overview.get('tags', []),  # Check if 'evasive' tag exists
#                    "scanner_crowdstrike_percent": scanners_v2.get('crowdstrike_ml', {}).get('percent', None),
#                    "scanner_metadefender_percent": scanners_v2.get('metadefender', {}).get('percent', None),
#                    "scanner_metadefender_positives": scanners_v2.get('metadefender', {}).get('positives', None),
                    #"multiscan_result": overview.get('multiscan_result')

        "is_peexe": int("peexe" in overview.get("type_short", [])),
        "is_64bit": int("64bits" in overview.get("type_short", [])),

        "is_executable": int("executable" in overview.get("type_short", [])),
        "tag_evasive": int("evasive" in overview.get("tags", [])),
         "scanner_crowdstrike_percent": (scanners_v2.get('crowdstrike_ml') or {}).get('percent'),
                "scanner_metadefender_percent": (scanners_v2.get('metadefender') or {}).get('percent'),
                "scanner_metadefender_positives": (scanners_v2.get('metadefender') or {}).get('positives'),
                                                                                              
        "multiscan_result": overview.get("multiscan_result", 0)
                }
                overview_data.append(row)

    # Write the overview data to CSV
    write_to_csv(overview_data)

if __name__ == "__main__":
    process_data()

