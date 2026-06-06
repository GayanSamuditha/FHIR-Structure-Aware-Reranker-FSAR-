"""Render FHIR resources into natural language statements."""


def render_resource(resource: dict, date: str | None, codes: list[dict]) -> str:
    """
    Convert a FHIR resource into a compact, structured, date-stamped statement.

    Args:
        resource: The FHIR resource dict
        date: The primary clinical date (ISO 8601) or None
        codes: List of code dicts [{system, code, display}]

    Returns:
        A formatted statement string

    Examples:
        - Observation: "2019-03-14 | lab | Serum creatinine 1.8 mg/dL (LOINC 2160-0)"
        - Condition: "2017-06-02 | diagnosis | Type 2 diabetes mellitus (SNOMED 44054006)"
        - MedicationRequest: "2019-04-01 | medication started | Metformin 500mg (RxNorm 860975)"
        - Procedure: "2018-11-20 | procedure | Echocardiography (SNOMED 40701008)"
    """
    resource_type = resource.get("resourceType", "Unknown")

    # Extract date portion (just the date, not time)
    date_str = date.split("T")[0] if date else "unknown-date"

    # Get the primary display text and code
    code_info = _get_primary_code_info(codes)
    description = _build_description(resource, resource_type, code_info)

    # Get category/type label
    category = _get_category(resource_type)

    # Format: "date | category | description"
    return f"{date_str} | {category} | {description}"


def _get_primary_code_info(codes: list[dict]) -> dict:
    """Extract the primary code display and system info."""
    if not codes:
        return {"display": None, "code": None, "system": None}

    # Prefer codes in this order: SNOMED, LOINC, RxNorm, others
    priority_systems = [
        "http://snomed.info/sct",
        "http://loinc.org",
        "http://www.nlm.nih.gov/research/umls/rxnorm",
    ]

    # Try priority systems first
    for system in priority_systems:
        for code in codes:
            if code.get("system") == system:
                return {
                    "display": code.get("display", ""),
                    "code": code.get("code", ""),
                    "system": _shorten_system_name(system),
                }

    # Fall back to first code
    first_code = codes[0]
    return {
        "display": first_code.get("display", ""),
        "code": first_code.get("code", ""),
        "system": _shorten_system_name(first_code.get("system", "")),
    }


def _shorten_system_name(system: str) -> str:
    """Convert full system URL to short name."""
    if "snomed" in system.lower():
        return "SNOMED"
    elif "loinc" in system.lower():
        return "LOINC"
    elif "rxnorm" in system.lower():
        return "RxNorm"
    else:
        return "code"


def _build_description(resource: dict, resource_type: str, code_info: dict) -> str:
    """Build the description part of the statement."""
    display = code_info.get("display", "")
    code = code_info.get("code", "")
    system = code_info.get("system", "")

    # Base description from code display
    desc_parts = []

    if resource_type == "Observation":
        # Try to include value if present
        value_str = _extract_observation_value(resource)
        if value_str and display:
            desc_parts.append(f"{display} {value_str}")
        elif display:
            desc_parts.append(display)
        else:
            desc_parts.append("observation")

    elif resource_type == "MedicationRequest":
        # Include dosage if available
        dosage_str = _extract_dosage(resource)
        if dosage_str and display:
            desc_parts.append(f"{display} {dosage_str}")
        elif display:
            desc_parts.append(display)
        else:
            desc_parts.append("medication")

    elif resource_type == "Condition":
        if display:
            desc_parts.append(display)
        else:
            desc_parts.append("condition")

    elif resource_type == "Procedure":
        if display:
            desc_parts.append(display)
        else:
            desc_parts.append("procedure")

    elif resource_type == "Encounter":
        # Use encounter type or class
        encounter_type = _extract_encounter_type(resource)
        if encounter_type:
            desc_parts.append(encounter_type)
        else:
            desc_parts.append("encounter")

    elif resource_type == "DiagnosticReport":
        if display:
            desc_parts.append(display)
        else:
            desc_parts.append("diagnostic report")
    else:
        desc_parts.append(display if display else resource_type.lower())

    description = " ".join(desc_parts) if desc_parts else resource_type.lower()

    # Add code reference if available
    if code and system:
        description += f" ({system} {code})"

    return description


def _extract_observation_value(resource: dict) -> str:
    """Extract value from an Observation resource."""
    # Try valueQuantity first (most common for labs)
    if "valueQuantity" in resource:
        value = resource["valueQuantity"].get("value", "")
        unit = resource["valueQuantity"].get("unit", "")
        if value and unit:
            return f"{value} {unit}"
        elif value:
            return str(value)

    # Try valueString
    if "valueString" in resource:
        return resource["valueString"]

    # Try valueCodeableConcept
    if "valueCodeableConcept" in resource:
        coding = resource["valueCodeableConcept"].get("coding", [])
        if coding and coding[0].get("display"):
            return coding[0]["display"]

    return ""


def _extract_dosage(resource: dict) -> str:
    """Extract dosage information from MedicationRequest."""
    if "dosageInstruction" in resource and resource["dosageInstruction"]:
        dosage = resource["dosageInstruction"][0]
        if "doseAndRate" in dosage and dosage["doseAndRate"]:
            dose_and_rate = dosage["doseAndRate"][0]
            if "doseQuantity" in dose_and_rate:
                dose_qty = dose_and_rate["doseQuantity"]
                value = dose_qty.get("value", "")
                unit = dose_qty.get("unit", "")
                if value and unit:
                    return f"{value}{unit}"
                elif value:
                    return str(value)
    return ""


def _extract_encounter_type(resource: dict) -> str:
    """Extract encounter type or class."""
    # Try type first
    if "type" in resource and resource["type"]:
        type_cc = resource["type"][0]
        if "coding" in type_cc and type_cc["coding"]:
            display = type_cc["coding"][0].get("display", "")
            if display:
                return display

    # Fall back to class
    if "class" in resource:
        class_coding = resource["class"]
        if "display" in class_coding:
            return class_coding["display"]
        elif "code" in class_coding:
            return class_coding["code"]

    return ""


def _get_category(resource_type: str) -> str:
    """Get the category label for a resource type."""
    categories = {
        "Observation": "lab",
        "Condition": "diagnosis",
        "MedicationRequest": "medication started",
        "Procedure": "procedure",
        "Encounter": "encounter",
        "DiagnosticReport": "diagnostic report",
    }
    return categories.get(resource_type, resource_type.lower())
