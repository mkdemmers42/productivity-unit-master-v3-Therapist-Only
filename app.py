import io
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# MASTER v3 Configuration
# -----------------------------
VALID_CODES = {
    "TCM/ICC",
    "Psychosocial Rehab - Individual",
    "Psychosocial Rehabilitation Group",
    "Non-billable Attempted Contact",
    "Client Non Billable Srvc Must Document",
    "Crisis Intervention",
    "Plan Development, non-physician",
    "Brief Contact Note",
    "Targeted Outreach",
}

BILLABLE_FACE_TO_FACE_CODES = {
    "TCM/ICC",
    "Psychosocial Rehab - Individual",
    "Psychosocial Rehabilitation Group",
    "Crisis Intervention",
    "Plan Development, non-physician",
    "Brief Contact Note",
    "Targeted Outreach",
}

NON_BILLABLE_FTF_CODES = {
    "Non-billable Attempted Contact",
    "Client Non Billable Srvc Must Document",
}

REQUIRED_COLS_CANONICAL = [
    "Procedure Code Name",
    "Travel Time",
    "Documentation Time",
    "Face-to-Face Time",
]

TOL_MINUTES_WORKED = 0.1
TOL_PERCENT = 0.01


# -----------------------------
# Data Structures
# -----------------------------
@dataclass(frozen=True)
class Results:
    hours_worked: float
    minutes_worked: float
    minutes_billed: int
    units_billed: int
    non_billable_total: int
    documentation_total: int
    travel_total: int
    billable_minutes_pct: float
    billable_units_pct: float
    non_billable_pct: float
    documentation_pct: float
    travel_pct: float


# -----------------------------
# Utility functions
# -----------------------------
def normalize_header(value: Any) -> str:
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s.strip()


def canonicalize_headers(cols: List[Any]) -> Dict[Any, str]:
    """
    MASTER v3 Header Normalization:
    - Trim whitespace
    - Convert line breaks to spaces
    - Standardize face-to-face variants to canonical
    """
    mapping: Dict[Any, str] = {}
    for c in cols:
        n = normalize_header(c)
        low = n.lower()

        if low == "procedure code name":
            mapping[c] = "Procedure Code Name"
            continue

        if low.replace("-", " ") == "travel time":
            mapping[c] = "Travel Time"
            continue

        if low.replace("-", " ") == "documentation time":
            mapping[c] = "Documentation Time"
            continue

        # Face-to-face variants
        ftf = low.replace("face to face", "face-to-face")
        ftf = ftf.replace("–", "-").replace("—", "-")
        if ftf == "face-to-face time":
            mapping[c] = "Face-to-Face Time"
            continue

        mapping[c] = n
    return mapping


# -----------------------------
# NEW: Auto-header detection (fixes "blank row above headers" issue)
# -----------------------------
def find_header_row_index_0_based(file_bytes: bytes, scan_rows: int = 40) -> int:
    """
    Finds the header row by scanning the first N rows for the presence of
    'Procedure Code Name' (normalized). Returns 0-based row index.
    """
    bio = io.BytesIO(file_bytes)
    preview = pd.read_excel(bio, header=None, nrows=scan_rows, dtype=object)

    for i in range(len(preview)):
        row = preview.iloc[i].tolist()
        normalized = [normalize_header(x).lower() for x in row]
        if "procedure code name" in normalized:
            return i

    raise ValueError(
        "Could not locate the header row. Expected to find 'Procedure Code Name' "
        "within the first rows of the spreadsheet."
    )


def load_excel_auto_header(file_bytes: bytes, dtype=object) -> Tuple[pd.DataFrame, int]:
    """
    Loads Excel using auto-detected header row. Returns (df, header_row_index_0_based).
    """
    header_idx = find_header_row_index_0_based(file_bytes)

    bio = io.BytesIO(file_bytes)
    df = pd.read_excel(bio, header=header_idx, dtype=dtype)
    return df, header_idx


def unit_grid(minutes: float) -> int:
    """
    MASTER v3 unit grid. Ceiling at 16 for >248.
    """
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        m = 0.0
    else:
        m = float(minutes)

    if m <= 7:
        return 0
    if m <= 22:
        return 1
    if m <= 37:
        return 2
    if m <= 52:
        return 3
    if m <= 67:
        return 4
    if m <= 82:
        return 5
    if m <= 97:
        return 6
    if m <= 112:
        return 7
    if m <= 127:
        return 8
    if m <= 142:
        return 9
    if m <= 157:
        return 10
    if m <= 172:
        return 11
    if m <= 187:
        return 12
    if m <= 202:
        return 13
    if m <= 217:
        return 14
    if m <= 232:
        return 15
    return 16


def round_minutes_worked(m: float) -> float:
    return round(m, 1)


def round_pct(p: float) -> float:
    return round(p, 2)


def compute_pass(
    hours_worked: float,
    file_bytes: bytes,
    audit: Optional[Dict[str, Any]] = None,
) -> Results:
    """
    FULL WORKFLOW PASS: load -> clean -> compute
    Header now auto-detected (supports both clean exports and ones with blank rows).
    Hidden math: return results only; optionally record audit details.
    """
    minutes_worked_raw = hours_worked * 60.0

    # Load (AUTO header detection)
    df, header_idx = load_excel_auto_header(file_bytes, dtype=object)

    # Normalize / canonicalize headers
    original_cols = list(df.columns)
    mapping = canonicalize_headers(original_cols)
    df = df.rename(columns=mapping)

    # Confirm required columns exist
    for col in REQUIRED_COLS_CANONICAL:
        if col not in df.columns:
            raise ValueError(f"MISSING REQUIRED COLUMN: {col}")

    # Clean Procedure Code Name
    df["Procedure Code Name"] = df["Procedure Code Name"].astype(str).str.strip()

    # Exclude rows containing "total"
    df = df[~df["Procedure Code Name"].str.contains("total", case=False, na=False)].copy()

    # Numeric coercion + zero fill (NO EXCEPTIONS)
    minute_cols = ["Travel Time", "Documentation Time", "Face-to-Face Time"]
    for c in minute_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Procedure code validation
    invalid = sorted(set(df["Procedure Code Name"].unique()) - VALID_CODES)
    if invalid:
        raise ValueError("INVALID PROCEDURE CODE(S) FOUND:\n" + "\n".join(invalid))

    # Totals
    non_billable_total = int(
        df.loc[df["Procedure Code Name"].isin(NON_BILLABLE_FTF_CODES), "Face-to-Face Time"].sum()
    )
    documentation_total = int(df["Documentation Time"].sum())
    travel_total = int(df["Travel Time"].sum())
    minutes_billed = int(
        df.loc[df["Procedure Code Name"].isin(BILLABLE_FACE_TO_FACE_CODES), "Face-to-Face Time"].sum()
    )

    # Units billed: apply grid per entry on billable rows; sum
    billable_ftf = df.loc[df["Procedure Code Name"].isin(BILLABLE_FACE_TO_FACE_CODES), "Face-to-Face Time"]
    units_billed = int(billable_ftf.apply(unit_grid).sum())

    # Percentages
    if minutes_worked_raw == 0:
        billable_minutes_pct = 0.0
        billable_units_pct = 0.0
        non_billable_pct = 0.0
        documentation_pct = 0.0
        travel_pct = 0.0
    else:
        billable_minutes_pct = (minutes_billed / minutes_worked_raw) * 100.0
        billable_units_pct = ((units_billed * 15.0) / minutes_worked_raw) * 100.0
        non_billable_pct = (non_billable_total / minutes_worked_raw) * 100.0
        documentation_pct = (documentation_total / minutes_worked_raw) * 100.0
        travel_pct = (travel_total / minutes_worked_raw) * 100.0

    res = Results(
        hours_worked=float(hours_worked),
        minutes_worked=round_minutes_worked(minutes_worked_raw),
        minutes_billed=minutes_billed,
        units_billed=units_billed,
        non_billable_total=non_billable_total,
        documentation_total=documentation_total,
        travel_total=travel_total,
        billable_minutes_pct=round_pct(billable_minutes_pct),
        billable_units_pct=round_pct(billable_units_pct),
        non_billable_pct=round_pct(non_billable_pct),
        documentation_pct=round_pct(documentation_pct),
        travel_pct=round_pct(travel_pct),
    )

    # Audit (saved only for download, not displayed)
    if audit is not None:
        audit["header_row_1_indexed"] = int(header_idx + 1)
        audit["original_columns"] = [str(c) for c in original_cols]
        audit["renamed_columns"] = list(df.columns)
        audit["row_count_after_clean"] = int(len(df))
        audit["unique_codes"] = sorted(df["Procedure Code Name"].unique().tolist())
        audit["intermediate"] = {
            "minutes_worked_raw": minutes_worked_raw,
            "minutes_billed": minutes_billed,
            "units_billed": units_billed,
            "non_billable_total": non_billable_total,
            "documentation_total": documentation_total,
            "travel_total": travel_total,
            "billable_minutes_pct_raw": billable_minutes_pct,
            "billable_units_pct_raw": billable_units_pct,
            "non_billable_pct_raw": non_billable_pct,
            "documentation_pct_raw": documentation_pct,
            "travel_pct_raw": travel_pct,
        }

    return res


def compare_results(p1: Results, p2: Results) -> Tuple[bool, List[str]]:
    mismatches: List[str] = []

    def mm(name: str, a: Any, b: Any) -> None:
        mismatches.append(f"{name}: Pass1={a} Pass2={b}")

    if p1.hours_worked != p2.hours_worked:
        mm("Hours_Worked", p1.hours_worked, p2.hours_worked)

    if abs(p1.minutes_worked - p2.minutes_worked) > TOL_MINUTES_WORKED:
        mm("Minutes_Worked", p1.minutes_worked, p2.minutes_worked)

    # exact match requirements
    if p1.minutes_billed != p2.minutes_billed:
        mm("Minutes_Billed", p1.minutes_billed, p2.minutes_billed)
    if p1.units_billed != p2.units_billed:
        mm("Units_Billed", p1.units_billed, p2.units_billed)
    if p1.non_billable_total != p2.non_billable_total:
        mm("Non_Billable_Total", p1.non_billable_total, p2.non_billable_total)
    if p1.documentation_total != p2.documentation_total:
        mm("Documentation_Time_Total", p1.documentation_total, p2.documentation_total)
    if p1.travel_total != p2.travel_total:
        mm("Travel_Time_Total", p1.travel_total, p2.travel_total)

    # percentage tolerance
    pct_fields = [
        ("Billable_Minutes_Percentage", p1.billable_minutes_pct, p2.billable_minutes_pct),
        ("Billable_Units_Percentage", p1.billable_units_pct, p2.billable_units_pct),
        ("Non_Billable_Percentage", p1.non_billable_pct, p2.non_billable_pct),
        ("Documentation_Percentage", p1.documentation_pct, p2.documentation_pct),
        ("Travel_Percentage", p1.travel_pct, p2.travel_pct),
    ]
    for name, a, b in pct_fields:
        if abs(a - b) > TOL_PERCENT:
            mm(name, a, b)

    return (len(mismatches) == 0, mismatches)


def print_final(res: Results) -> None:
    st.success("VERIFICATION PASSED ✅")

    st.write(f"**Hours Worked:** {res.hours_worked}")
    st.write(f"**Minutes Worked:** {res.minutes_worked}")

    st.write("")
    st.write(f"**Minutes Billed:** {res.minutes_billed}")
    st.write(f"**Billable Minutes Percentage:** {res.billable_minutes_pct}%")

    st.write("")
    st.write(f"**Units Billed:** {res.units_billed}")
    st.write(f"**Billable Units Percentage:** {res.billable_units_pct}%")

    st.write("")
    st.write(f"**Non-Billable Total:** {res.non_billable_total}")
    st.write(f"**Non-Billable Percentage:** {res.non_billable_pct}%")

    st.write("")
    st.write(f"**Documentation Time Total:** {res.documentation_total}")
    st.write(f"**Documentation Percentage:** {res.documentation_pct}%")

    st.write("")
    st.write(f"**Travel Time Total:** {res.travel_total}")
    st.write(f"**Travel Percentage:** {res.travel_pct}%")


# -----------------------------
# Streamlit UI (Hidden Math)
# -----------------------------
st.set_page_config(page_title="Mike's Productivity/Unit Machine - Case Managers Only", layout="centered")
st.title("Mike's Productivity/Unit Machine (v3) - Case Managers Only")


# Session init
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_audit_payload" not in st.session_state:
    st.session_state["last_audit_payload"] = None
if "last_error" not in st.session_state:
    st.session_state["last_error"] = None
if "reset_counter" not in st.session_state:
    st.session_state["reset_counter"] = 0


def do_reset() -> None:
    # Changing widget keys is the reliable way to clear text_input + file_uploader
    st.session_state["reset_counter"] += 1
    st.session_state["last_result"] = None
    st.session_state["last_audit_payload"] = None
    st.session_state["last_error"] = None
    st.rerun()


# Widget keys that change after each reset
k = st.session_state["reset_counter"]
hours_key = f"hours_{k}"
file_key = f"uploaded_file_{k}"

hours = st.text_input("Please insert **Hours Worked**", placeholder="Example: 148.13", key=hours_key)

st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)

uploaded = st.file_uploader(
    'Upload the **Excel (.xlsx) file** exported from **"Staff Service Detail Report"** from SmartCare',
    type=["xlsx"],
    key=file_key
)

col_run, col_reset = st.columns([1, 1])
with col_run:
    run = st.button("Calculate Productivity", type="primary")
with col_reset:
    reset = st.button("Run Another Staff Member", on_click=do_reset)

st.divider()


if run:
    st.session_state["last_error"] = None

    # Hours validation (failsafe)
    try:
        hours_worked = float((hours or "").strip())
    except Exception:
        st.session_state["last_error"] = "Please enter Hours Worked as a number only (example: 128.4)."
        st.stop()

    if uploaded is None:
        st.session_state["last_error"] = (
            "Please upload the Excel spreadsheet containing staff's 'Billed' and 'Non-Billable' numbers."
        )
        st.stop()

    file_bytes = uploaded.getvalue()

    # PASS 1
    audit1: Dict[str, Any] = {}
    try:
        pass1 = compute_pass(hours_worked, file_bytes, audit=audit1)
    except ValueError as e:
        st.session_state["last_error"] = str(e)
        st.stop()
    except Exception as e:
        st.session_state["last_error"] = f"ERROR LOADING/PROCESSING FILE: {e}"
        st.stop()

    # PASS 2 (verification recompute from scratch)
    audit2: Dict[str, Any] = {}
    try:
        pass2 = compute_pass(hours_worked, file_bytes, audit=audit2)
    except Exception as e:
        st.session_state["last_error"] = f"VERIFICATION FAILED — RESULTS NOT TRUSTWORTHY\n\nReason: {e}"
        st.stop()

    ok, mismatches = compare_results(pass1, pass2)
    if not ok:
        st.session_state["last_error"] = (
            "VERIFICATION FAILED — RESULTS NOT TRUSTWORTHY\n\nMetric(s) mismatched:\n" + "\n".join(mismatches)
        )
        st.stop()

    # Store results for display
    st.session_state["last_result"] = pass1
    st.session_state["last_audit_payload"] = {
        "pass1": audit1,
        "pass2": audit2,
        "final": {
            "hours_worked": pass1.hours_worked,
            "minutes_worked": pass1.minutes_worked,
            "minutes_billed": pass1.minutes_billed,
            "units_billed": pass1.units_billed,
            "non_billable_total": pass1.non_billable_total,
            "documentation_total": pass1.documentation_total,
            "travel_total": pass1.travel_total,
            "billable_minutes_pct": pass1.billable_minutes_pct,
            "billable_units_pct": pass1.billable_units_pct,
            "non_billable_pct": pass1.non_billable_pct,
            "documentation_pct": pass1.documentation_pct,
            "travel_pct": pass1.travel_pct,
        },
    }
    st.rerun()


# Display
if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])

if st.session_state["last_result"] is not None:
    print_final(st.session_state["last_result"])

    if st.session_state["last_audit_payload"] is not None:
        st.download_button(
            "Download Audit JSON (internal math, not displayed)",
            data=json.dumps(st.session_state["last_audit_payload"], indent=2).encode("utf-8"),
            file_name="productivity_audit.json",
            mime="application/json",
        )

    st.write("")
    st.info("Ready for the next staff member? Click **Run Another Staff Member**.")
