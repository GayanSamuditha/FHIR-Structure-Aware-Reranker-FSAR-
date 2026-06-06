"""Ingest Synthea FHIR bundles into canonical records and reference graph."""

import hashlib
import json
from pathlib import Path
from typing import Any
from collections import defaultdict

import pandas as pd

from src import config
from src.render import render_resource


def _generate_hash_id(patient_id: str, resource_type: str, resource: dict, index: int) -> str:
    """Generate a stable content-based ID for resources that have no id/fullUrl."""
    parts = [patient_id, resource_type, str(index)]
    for field in ("dateWritten", "performedDateTime", "onsetDateTime", "authoredOn", "effectiveDateTime"):
        val = resource.get(field)
        if val:
            parts.append(str(val))
            break
    for cc_field in ("medicationCodeableConcept", "code"):
        cc = resource.get(cc_field) or {}
        coding = cc.get("coding", []) if isinstance(cc, dict) else []
        if coding:
            parts.append(coding[0].get("code", ""))
            break
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def load_bundles(limit: int | None = None) -> list[dict]:
    """
    Load FHIR bundles from the data directory.

    Args:
        limit: Maximum number of bundles to load (for testing)

    Returns:
        List of bundle dictionaries
    """
    bundles = []
    data_dir = Path(config.DATA_DIR)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Get all JSON files recursively, excluding hospital/practitioner information
    json_files = []
    for json_file in sorted(data_dir.glob("**/*.json")):
        # Skip files matching patterns in config
        skip = False
        for pattern in config.SKIP_FILE_PATTERNS:
            if pattern in json_file.name:
                skip = True
                break
        if not skip:
            json_files.append(json_file)

    # Apply limit if specified
    if limit:
        json_files = json_files[:limit]

    # Load bundles
    for json_file in json_files:
        with open(json_file, "r") as f:
            bundle = json.load(f)
            bundles.append(bundle)

    return bundles


def build_reference_map(bundle: dict) -> dict[str, tuple[str, str]]:
    """
    Build a mapping from fullUrl to (resourceType, id) for a bundle.

    Args:
        bundle: A FHIR bundle

    Returns:
        Dictionary mapping fullUrl to (resourceType, id)
    """
    ref_map = {}
    for entry in bundle.get("entry", []):
        full_url = entry.get("fullUrl", "")
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "")
        resource_id = resource.get("id", "")

        if full_url and resource_type:
            ref_map[full_url] = (resource_type, resource_id)

    return ref_map


def resolve_reference(ref_string: str, ref_map: dict[str, tuple[str, str]]) -> str | None:
    """
    Resolve a FHIR reference string to a canonical ID.

    Args:
        ref_string: Reference string (e.g., "urn:uuid:..." or "ResourceType/id")
        ref_map: Mapping from fullUrl to (resourceType, id)

    Returns:
        Canonical ID in format "ResourceType/id" or None if not resolved
    """
    if not ref_string:
        return None

    # Direct format: "ResourceType/id"
    if "/" in ref_string and not ref_string.startswith("urn:"):
        return ref_string

    # URN format: "urn:uuid:..."
    if ref_string in ref_map:
        resource_type, resource_id = ref_map[ref_string]
        return f"{resource_type}/{resource_id}"

    return None


def extract_date(resource: dict) -> str | None:
    """
    Extract the primary clinical date from a resource.

    Args:
        resource: A FHIR resource

    Returns:
        ISO 8601 date string or None
    """
    resource_type = resource.get("resourceType", "")
    date_field = config.DATE_FIELDS.get(resource_type)

    if not date_field:
        return None

    # Handle list of possible fields (e.g., Procedure)
    if isinstance(date_field, list):
        for field in date_field:
            date_value = _get_nested_field(resource, field)
            if date_value:
                return date_value
    else:
        date_value = _get_nested_field(resource, date_field)
        if date_value:
            return date_value

    # FHIR STU3 fallback: MedicationRequest uses dateWritten instead of authoredOn
    if resource_type == "MedicationRequest":
        return resource.get("dateWritten")

    return None


def _get_nested_field(obj: dict, field_path: str) -> Any:
    """Get a nested field using dot notation (e.g., 'period.start')."""
    parts = field_path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def extract_codes(resource: dict) -> list[dict]:
    """
    Extract coding information from a resource.

    Args:
        resource: A FHIR resource

    Returns:
        List of code dictionaries [{system, code, display}]
    """
    resource_type = resource.get("resourceType", "")
    codes = []

    # Get the code field name for this resource type
    code_field_name = config.CODE_FIELDS.get(resource_type, "code")

    # Get the CodeableConcept
    code_concept = resource.get(code_field_name)
    if not code_concept:
        return codes

    # Extract coding array
    coding = code_concept.get("coding", [])
    for code_entry in coding:
        codes.append({
            "system": code_entry.get("system", ""),
            "code": code_entry.get("code", ""),
            "display": code_entry.get("display", ""),
        })

    return codes


def extract_references(resource: dict, ref_map: dict[str, tuple[str, str]]) -> tuple[str | None, list[str]]:
    """
    Extract encounter_id and all references from a resource.

    Args:
        resource: A FHIR resource
        ref_map: Reference mapping for resolving URN references

    Returns:
        Tuple of (encounter_id, list of reference IDs)
    """
    encounter_id = None
    references = []

    # Extract encounter reference (FHIR R4: 'encounter'; FHIR STU3 MedicationRequest: 'context')
    for enc_field in ("encounter", "context"):
        enc_obj = resource.get(enc_field)
        if isinstance(enc_obj, dict):
            encounter_ref = enc_obj.get("reference", "")
            resolved = resolve_reference(encounter_ref, ref_map)
            if resolved and resolved.startswith("Encounter/"):
                encounter_id = resolved
                references.append(resolved)
                break

    # Extract other references
    for field in config.REFERENCE_FIELDS:
        if field == "encounter":
            continue  # Already handled

        if field in resource:
            field_value = resource[field]

            # Handle reference object
            if isinstance(field_value, dict) and "reference" in field_value:
                ref_string = field_value["reference"]
                resolved = resolve_reference(ref_string, ref_map)
                if resolved:
                    references.append(resolved)

            # Handle array of references (e.g., result[])
            elif isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, dict) and "reference" in item:
                        ref_string = item["reference"]
                        resolved = resolve_reference(ref_string, ref_map)
                        if resolved:
                            references.append(resolved)

    return encounter_id, references


def process_bundle(bundle: dict) -> list[dict]:
    """
    Process a single FHIR bundle into canonical records.

    Args:
        bundle: A FHIR bundle

    Returns:
        List of record dictionaries
    """
    records = []
    ref_map = build_reference_map(bundle)

    # Find patient ID
    patient_id = None
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            patient_id = resource.get("id", "")
            break

    if not patient_id:
        # Skip bundles without a patient
        return records

    # Process each entry
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType", "")

        # Only process configured resource types
        if resource_type not in config.RESOURCE_TYPES:
            continue

        # Derive stable resource_id: prefer resource.id, then fullUrl UUID, then content hash
        resource_id = resource.get("id", "")
        if not resource_id:
            full_url = entry.get("fullUrl", "") or ""
            if full_url.startswith("urn:uuid:"):
                resource_id = full_url[len("urn:uuid:"):]
            else:
                resource_id = _generate_hash_id(patient_id, resource_type, resource, len(records))

        # Extract fields
        date = extract_date(resource)
        codes = extract_codes(resource)
        encounter_id, references = extract_references(resource, ref_map)

        # Generate statement
        text = render_resource(resource, date, codes)

        # Create canonical record
        record = {
            "id": f"{resource_type}/{resource_id}",
            "patient_id": patient_id,
            "resource_type": resource_type,
            "text": text,
            "date": date,
            "codes": codes,
            "encounter_id": encounter_id,
            "references": references,
        }

        records.append(record)

    return records


def build_reference_graph(records: list[dict]) -> dict:
    """
    Build the reference graph from canonical records.

    Args:
        records: List of canonical record dictionaries

    Returns:
        Dictionary with by_encounter and adjacency mappings
    """
    by_encounter = defaultdict(list)
    adjacency = defaultdict(set)

    for record in records:
        record_id = record["id"]
        encounter_id = record["encounter_id"]
        references = record["references"]

        # Group by encounter
        if encounter_id:
            by_encounter[encounter_id].append(record_id)

        # Build adjacency (bidirectional)
        for ref_id in references:
            # Outgoing edge
            adjacency[record_id].add(ref_id)
            # Incoming edge
            adjacency[ref_id].add(record_id)

    # Convert sets to lists for JSON serialization
    adjacency_dict = {k: list(v) for k, v in adjacency.items()}
    by_encounter_dict = dict(by_encounter)

    return {
        "by_encounter": by_encounter_dict,
        "adjacency": adjacency_dict,
    }


def ingest_all(limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Ingest all FHIR bundles and produce statements + reference graph.

    Args:
        limit: Maximum number of bundles to process

    Returns:
        Tuple of (statements DataFrame, reference graph dict)
    """
    print(f"Loading bundles from {config.DATA_DIR}...")
    bundles = load_bundles(limit=limit or config.SUBSET_PATIENTS)
    print(f"Loaded {len(bundles)} bundles")

    print("Processing bundles...")
    all_records = []
    for i, bundle in enumerate(bundles):
        records = process_bundle(bundle)
        all_records.extend(records)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(bundles)} bundles, {len(all_records)} records so far")

    print(f"Processed {len(bundles)} bundles -> {len(all_records)} records")

    # Convert to DataFrame
    df = pd.DataFrame(all_records)

    # Build reference graph
    print("Building reference graph...")
    refgraph = build_reference_graph(all_records)
    print(f"  {len(refgraph['by_encounter'])} encounters")
    print(f"  {len(refgraph['adjacency'])} nodes in adjacency graph")

    return df, refgraph


def save_artifacts(df: pd.DataFrame, refgraph: dict) -> None:
    """
    Save statements and reference graph to disk.

    Args:
        df: Statements DataFrame
        refgraph: Reference graph dictionary
    """
    print(f"Writing {len(df)} statements to {config.STATEMENTS_PATH}...")
    df.to_parquet(config.STATEMENTS_PATH, index=False)

    print(f"Writing reference graph to {config.REFGRAPH_PATH}...")
    with open(config.REFGRAPH_PATH, "w") as f:
        json.dump(refgraph, f, indent=2)

    print("✓ Artifacts saved")


def main():
    """Main entry point for M1 ingestion."""
    df, refgraph = ingest_all()
    save_artifacts(df, refgraph)

    # Print summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Resource types: {df['resource_type'].value_counts().to_dict()}")
    print(f"Patients: {df['patient_id'].nunique()}")
    print(f"Encounters: {len(refgraph['by_encounter'])}")
    print(f"Reference graph nodes: {len(refgraph['adjacency'])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
