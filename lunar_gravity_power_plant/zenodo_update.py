#!/usr/bin/env python3
"""Upload v3 of the lunar gravity power plant paper as a new Zenodo version."""

import json
import os
import sys
import requests

ZENODO_TOKEN = os.environ.get("ZENODO_API_KEY", "") or os.environ.get("ZENODO_TOKEN", "")
BASE_URL = "https://zenodo.org/api"

# The existing record's concept ID (the DOI-all version)
# From https://doi.org/10.5281/zenodo.19177649
# You may need to adjust this — it's the numeric ID of your existing deposit
RECORD_ID = 19188043  # actual record ID (concept ID is 19177649)

PDF_PATH = "/home/jonsmirl/materials/lunar_gravity_power_plant/Lunar_Gravity_Power_Plant.pdf"


def update_paper(token, base_url, record_id, pdf_path):
    headers = {"Authorization": f"Bearer {token}"}

    # Step 1: Create new version of existing record
    print(f"Creating new version of record {record_id}...")
    r = requests.post(
        f"{base_url}/deposit/depositions/{record_id}/actions/newversion",
        headers=headers,
    )
    if r.status_code != 201:
        print(f"  ERROR creating new version: {r.status_code} {r.text}")
        print(f"  If record ID is wrong, check your Zenodo deposit page for the numeric ID.")
        return None

    # Get the new draft's deposit URL
    new_version = r.json()
    draft_url = new_version["links"]["latest_draft"]
    print(f"  New version draft: {draft_url}")

    # Step 2: Get the new draft details
    r = requests.get(draft_url, headers=headers)
    if r.status_code != 200:
        print(f"  ERROR getting draft: {r.status_code} {r.text}")
        return None
    draft = r.json()
    new_id = draft["id"]
    bucket_url = draft["links"]["bucket"]
    print(f"  New deposit ID: {new_id}")

    # Step 3: Delete old files from the new draft
    for f in draft.get("files", []):
        file_id = f["id"]
        r = requests.delete(
            f"{base_url}/deposit/depositions/{new_id}/files/{file_id}",
            headers=headers,
        )
        print(f"  Deleted old file: {f['filename']}")

    # Step 4: Upload new PDF
    filename = os.path.basename(pdf_path)
    print(f"  Uploading {filename}...")
    with open(pdf_path, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{filename}",
            headers=headers,
            data=f,
        )
    if r.status_code not in (200, 201):
        print(f"  ERROR uploading: {r.status_code} {r.text}")
        return None
    print(f"  Uploaded {filename}")

    # Step 5: Update metadata for v3
    metadata = {
        "metadata": {
            "title": "A Practical Path to O'Neill Colonies: Bootstrapping Lunar Industry from a Superconducting Cable and an Asteroid",
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": (
                "v3: Superconductor changed from MgB₂ (Tc=39K) to Bi-2223 (Tc=110K) to solve "
                "solar thermal management of the 62,000 km cable in direct sunlight. New section: "
                "Solar Thermal Budget with 12 pitch-carbon-fiber radiating fins, including fundamental "
                "scaling law T⁴ = 2α·cos(π/N)·S/(ε·N·σ), fin efficiency analysis, eddy current "
                "compatibility proof (carbon fiber vs aluminum), and coating degradation margins. "
                "Cost updated from $40B to $48B. Cable mass from 108,000 to 124,000 tons. "
                "All physics validated; no change to the core gravity power plant concept.\n\n"
                "Building O'Neill-class space habitats requires gigawatt-scale power and millions of "
                "tons of steel — neither available off Earth. This paper presents a gravity power "
                "plant that provides both from a single system: M-type asteroid rock descends a "
                "62,000 km superconducting cable from Earth-Moon L1 to the lunar surface, generating "
                "1 GW at baseline throughput via Lenz's law braking. The rock — already metallic "
                "iron-nickel — is smelted into structural steel using the power it just generated. "
                "At $48B for 1 GW (vs $50-70B for lunar nuclear), this system provides a credible "
                "path to a self-sustaining industrial civilization across the solar system."
            ),
            "creators": [
                {
                    "name": "Smirl, Jon",
                    "affiliation": "Independent Researcher",
                }
            ],
            "keywords": [
                "space elevator", "lunar power", "O'Neill colony", "superconductor",
                "Bi-2223", "gravity power plant", "Lenz's law", "asteroid mining",
                "space habitat", "SMES", "radiating fins", "solar thermal management"
            ],
            "access_right": "open",
            "license": "cc-by-4.0",
            "version": "v3",
        }
    }
    r = requests.put(
        f"{base_url}/deposit/depositions/{new_id}",
        headers={**headers, "Content-Type": "application/json"},
        json=metadata,
    )
    if r.status_code != 200:
        print(f"  ERROR setting metadata: {r.status_code} {r.text}")
        return None
    print(f"  Metadata updated for v3")

    # Step 6: Publish
    r = requests.post(
        f"{base_url}/deposit/depositions/{new_id}/actions/publish",
        headers=headers,
    )
    if r.status_code != 202:
        print(f"  ERROR publishing: {r.status_code} {r.text}")
        print(f"  Draft saved — publish manually at: https://zenodo.org/deposit/{new_id}")
        return f"https://zenodo.org/deposit/{new_id}"

    result = r.json()
    doi = result.get("doi", "N/A")
    url = result.get("links", {}).get("html", f"https://zenodo.org/record/{new_id}")
    print(f"\n  PUBLISHED! DOI: {doi}")
    print(f"  URL: {url}")
    return url


def main():
    token = ZENODO_TOKEN
    if not token:
        token = input("Enter your Zenodo personal access token: ").strip()
    if not token:
        print("No token provided. Exiting.")
        sys.exit(1)

    if not os.path.exists(PDF_PATH):
        print(f"PDF not found: {PDF_PATH}")
        sys.exit(1)

    url = update_paper(token, BASE_URL, RECORD_ID, PDF_PATH)
    if url:
        print(f"\nDone. The DOI link will resolve to the new version automatically.")


if __name__ == "__main__":
    main()
