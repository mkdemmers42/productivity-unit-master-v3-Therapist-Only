import io
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# -----------------------------
# MASTER v3 Configuration (Merged Codes + Multi-Grid Billing)
# -----------------------------

# All known codes (merged, no duplicates because sets)
VALID_CODES = {
    # Case manager / rehab / outreach universe
    "TCM/ICC",
    "Psychosocial Rehab - Individual",
    "Psychosocial Rehabilitation Group",
    "Brief Contact Note",
    "Targeted Outreach",

    # Therapist / clinical universe
    "Individual Therapy",
    "Family Therapy",
    "Family Therapy - client present",
    "Assessment LPHA",
    "Crisis Intervention",
    "Plan Development, non-physician",

    # Non-billable
    "Non-billable Attempted Contact",
    "Client Non Billable Srvc Must Document",
}

# Non-billable codes (always 0 units)
NON_BILLABLE_CODES = {
    "Non-billable Attempted Contact",
    "Client Non Billable Srvc Must Document",
}

# Billable codes = everything valid except non-billable
BILLABLE_FACE_TO_FACE_CODES = VALID_CODES - NON_BILLABLE_CODES

REQUIRED_COLS_CANONICAL = [
    "Procedure Code Name",
    "Travel Time",
    "Documentation Time",
    "Face-to-Face Time",
]

TOL_MINUTES_WORKED = 0.1
TOL_PERCENT = 0.01


# -----------------------------
# Unit Grid Definitions (Per Your Rules)
# -----------------------------

def units_scale_b(minutes: float) -> int:
    """
    Scale B (Clinical grid):
    0–8 -> 0
    8–23 -> 1
    ...
    218–233 -> 15
    233+ -> 16
    """
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        m = 0.0
    else:
        m = float(minutes)

    if m <= 8:
        return 0
    if m <= 23:
        return 1
    if m <= 38:
        return 2
    if m <= 53:
        return 3
    if m <= 68:
        return 4
    if m <= 83:
        return 5
    if m <= 98:
        return 6
    if m <= 113:
        return 7
    if m <= 128:
        return 8
    if m <= 143:
        return 9
    if m <= 158:
        return 10
    if m <= 173:
        return 11
    if m <= 188:
        return 12
    if m <= 203:
        return 13
    if m <= 218:
        return 14
    if m <= 233:
        return 15
    return 16

def units_individual_therapy_scale_c(minutes: float) -> int:
    """
    Scale C (Individual Therapy) — Extended
    0–15 -> 0
    16–30 -> 2
    31–45 -> 3
    46–67 -> 4
    68–82 -> 5
    83–97 -> 6
    98–112 -> 7
    113–127 -> 8
    128–142 -> 9
    143–157 -> 10
    158–172 -> 11
    173–187 -> 12
    188–202 -> 13
    203–217 -> 14
    218–232 -> 15
    233+ -> 16
    """
    m = 0.0 if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)) else float(minutes)

    if m <= 15:
        return 0
    if m <= 30:
        return 2
    if m <= 45:
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



def units_assessment_lpha_scale_d(minutes: float) -> int:
    """
    Scale D (Assessment LPHA) — Extended
    0–30 -> 0
    31–45 -> 3
    46–67 -> 4
    68–82 -> 5
    83–97 -> 6
    98–112 -> 7
    113–127 -> 8
    128–142 -> 9
    143–157 -> 10
    158–172 -> 11
    173–187 -> 12
    188–202 -> 13
    203–217 -> 14
    218–232 -> 15
    233+ -> 16
    """
    m = 0.0 if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)) else float(minutes)

    if m <= 30:
        return 0
    if m <= 45:
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




def units_family_therapy_scale_e(minutes: float) -> int:
    """
    Scale E (Family Therapy) — Extended & Gap-Free

    0–26   -> 0
    27–57  -> 4
    58–72  -> 5
    73–87  -> 6
    88–102 -> 7
    103–117 -> 8
    118–132 -> 9
    133–147 -> 10
    148–162 -> 11
    163–177 -> 12
    178–192 -> 13
    193–207 -> 14
    208–222 -> 15
    223+    -> 16
    """

    m = 0.0 if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)) else float(minutes)

    if m <= 26:
        return 0
    if m <= 57:
        return 4
    if m <= 72:
        return 5
    if m <= 87:
        return 6
    if m <= 102:
        return 7
    if m <= 117:
        return 8
    if m <= 132:
        return 9
    if m <= 147:
        return 10
    if m <= 162:
        return 11
    if m <= 177:
        return 12
    if m <= 192:
        return 13
    if m <= 207:
        return 14
    if m <= 222:
        return 15
    return 16

    


CODE_TO_UNIT_FN = {

    # -------------------------
    # Scale B codes
    # -------------------------
    "TCM/ICC": units_scale_b,
    "Psychosocial Rehab - Individual": units_scale_b,
    "Psychosocial Rehabilitation Group": units_scale_b,
    "Crisis Intervention": units_scale_b,
    "Plan Development, non-physician": units_scale_b,
    "Brief Contact Note": units_scale_b,
    "Targeted Outreach": units_scale_b,

    # -------------------------
    # Scale C (Individual Therapy)
    # -------------------------
    "Individual Therapy": units_individual_therapy_scale_c,

    # -------------------------
    # Scale D (Assessment LPHA)
    # -------------------------
    "Assessment LPHA": units_assessment_lpha_scale_d,

    # -------------------------
    # Scale E (Family Therapy)
    # -------------------------
    "Family Therapy": units_family_therapy_scale_e,
    "Family Therapy - client present": units_family_therapy_scale_e,
}


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

        ftf = low.replace("face to face", "face-to-face")
        ftf = ftf.replace("–", "-").replace("—", "-")
        if ftf == "face-to-face time":
            mapping[c] = "Face-to-Face Time"
            continue

        mapping[c] = n
    return mapping


def find_header_row_index_0_based(file_bytes: bytes, scan_rows: int = 40) -> int:
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
    header_idx = find_header_row_index_0_based(file_bytes)
    bio = io.BytesIO(file_bytes)
    df = pd.read_excel(bio, header=header_idx, dtype=dtype)
    return df, header_idx


def round_minutes_worked(m: float) -> float:
    return round(m, 1)


def round_pct(p: float) -> float:
    return round(p, 2)


def units_for_row(code: str, minutes: float) -> int:
    """
    Returns units for a single row based on the code's assigned scale.
    Non-billable codes always return 0 units.
    """
    if code in NON_BILLABLE_CODES:
        return 0

    fn = CODE_TO_UNIT_FN.get(code)
    if fn is None:
        # If it's valid but not mapped, we should not guess.
        raise ValueError(
            f"UNIT SCALE NOT CONFIGURED for code: '{code}'. "
            "This code is in VALID_CODES but has no unit scale mapping."
        )
    return int(fn(minutes))


def compute_pass(
    hours_worked: float,
    file_bytes: bytes,
    audit: Optional[Dict[str, Any]] = None,
) -> Results:
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

    # Numeric coercion + zero fill
    minute_cols = ["Travel Time", "Documentation Time", "Face-to-Face Time"]
    for c in minute_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Procedure code validation
    invalid = sorted(set(df["Procedure Code Name"].unique()) - VALID_CODES)
    if invalid:
        raise ValueError("INVALID PROCEDURE CODE(S) FOUND:\n" + "\n".join(invalid))

    # Totals
    non_billable_total = int(
        df.loc[df["Procedure Code Name"].isin(NON_BILLABLE_CODES), "Face-to-Face Time"].sum()
    )
    documentation_total = int(df["Documentation Time"].sum())
    travel_total = int(df["Travel Time"].sum())

    minutes_billed = int(
        df.loc[df["Procedure Code Name"].isin(BILLABLE_FACE_TO_FACE_CODES), "Face-to-Face Time"].sum()
    )

    # Units billed: apply per-code unit mapping row-by-row
    billable_rows = df[df["Procedure Code Name"].isin(BILLABLE_FACE_TO_FACE_CODES)].copy()
    units_billed = int(
        billable_rows.apply(
            lambda r: units_for_row(str(r["Procedure Code Name"]), float(r["Face-to-Face Time"])),
            axis=1
        ).sum()
    )

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

        audit["code_to_scale"] = {
            "Scale B": sorted([c for c, fn in CODE_TO_UNIT_FN.items() if fn == units_scale_b]),
            "Scale C": ["Individual Therapy"],
            "Scale D": ["Assessment LPHA"],
            "Scale E": [
                "Family Therapy",
                "Family Therapy - client present",
            ],
            "Non-Billable": sorted(list(NON_BILLABLE_CODES)),
        }

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
st.set_page_config(page_title="Mike's Productivity/Unit Machine for Therapist", layout="centered")
st.title("Mike's Productivity/Unit Machine (v3) - Therapist Only")


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
    st.session_state["reset_counter"] += 1
    st.session_state["last_result"] = None
    st.session_state["last_audit_payload"] = None
    st.session_state["last_error"] = None


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


def fail(msg: str) -> None:
    msg = (msg or "").strip()
    if not msg.endswith("Tell Mike."):
        msg = f"{msg}\n\nTell Mike."
    st.session_state["last_error"] = msg
    st.error(msg)
    st.stop()


if run:
    st.session_state["last_error"] = None

    try:
        hours_worked = float((hours or "").strip())
    except Exception:
        fail("Please enter Hours Worked as a number only (example: 128.4).")

    if uploaded is None:
        fail("Please upload the Excel spreadsheet containing staff's 'Billed' and 'Non-Billable' numbers.")

    file_bytes = uploaded.getvalue()

    # PASS 1
    audit1: Dict[str, Any] = {}
    try:
        pass1 = compute_pass(hours_worked, file_bytes, audit=audit1)
    except ValueError as e:
        fail(str(e))
    except Exception as e:
        fail(f"ERROR LOADING/PROCESSING FILE: {e}")

    # PASS 2 (verification recompute from scratch)
    audit2: Dict[str, Any] = {}
    try:
        pass2 = compute_pass(hours_worked, file_bytes, audit=audit2)
    except Exception as e:
        fail(f"VERIFICATION FAILED — RESULTS NOT TRUSTWORTHY\n\nReason: {e}")

    ok, mismatches = compare_results(pass1, pass2)
    if not ok:
        fail(
            "VERIFICATION FAILED — RESULTS NOT TRUSTWORTHY\n\nMetric(s) mismatched:\n"
            + "\n".join(mismatches)
        )

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
