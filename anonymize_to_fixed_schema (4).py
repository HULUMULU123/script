#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anonymize (pseudonymize) PII in an Excel file for ML training and emit a fixed schema:
date, debit_account, debit_name, debit_inn, credit_account, credit_name, credit_inn,
debit_amount, credit_amount, purpose

Supports two input styles:
A) Separate columns for debit/credit account/name/inn (use --in-*-acct/--in-*-name/--in-*-inn)
B) Compound columns like "Дебет" / "Кредит" where a single cell contains multiple lines:
   Дебет
   11111111111111111111
   2222222222
   ООО "Шляпа"
   Provide with --in-debit and/or --in-credit.

- Deterministic HMAC-SHA256 with a secret salt for names/INN/accounts.
- Redaction of PII-like tokens inside "purpose".
- Optional fixed shift for transaction dates to keep seasonality but break real calendar alignment.
- Preserves numeric amounts; coerces to numbers where possible.
"""
import argparse
import hashlib
import hmac
import re
from datetime import timedelta
from typing import Optional, Dict, Tuple

import pandas as pd


# --------- HMAC helpers ---------
def hmac_hex(key: str, value: str, prefix: str = "", length: int = 16) -> str:
    """Return short deterministic hex token for a value using HMAC-SHA256."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    v = str(value).strip()
    if v == "":
        return v
    digest = hmac.new(key.encode("utf-8"), v.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{prefix}{digest[:length]}"


# --------- Text redaction in purpose ---------
DATE_RE = re.compile(r"\b(\d{2}\.\d{2}\.\d{4})\b")
INN_RE = re.compile(r"\b(\d{10}|\d{12})\b")  # 10/12 digits
ACC_RE = re.compile(r"\b(\d{20})\b")  # расчетный счет (20 digits)
CONTRACT_RE = re.compile(r"(?i)\bдог(ов|о)р(?:\s*№|\s*N|\s*№\s*|)\s*([A-Za-zА-Яа-я0-9/\-_.]+)")

def redact_purpose(text: str, key: str) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return text
    s = str(text)

    def repl_with(tag: str):
        def _repl(m: re.Match):
            token_src = m.group(0)
            token = hmac_hex(key, token_src, prefix=f"{tag}_", length=12)
            return token
        return _repl

    s = DATE_RE.sub(repl_with("DATE"), s)
    s = INN_RE.sub(repl_with("INN"), s)
    s = ACC_RE.sub(repl_with("ACC"), s)
    s = CONTRACT_RE.sub(lambda m: f"ДОГОВОР_{hmac_hex(key, m.group(0), prefix='', length=12)}", s)

    # common phone patterns
    s = re.sub(r"\b(?:\+7|8)\s*\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b",
               lambda m: f"PHONE_{hmac_hex(key, m.group(0), length=10)}", s)
    # e-mail
    s = re.sub(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b",
               lambda m: f"EMAIL_{hmac_hex(key, m.group(0), length=10)}", s)
    return s


def shift_date(val, days: int):
    if days == 0:
        return val
    try:
        ts = pd.to_datetime(val, dayfirst=True, errors="coerce")
        if pd.isna(ts):
            return val
        return ts + timedelta(days=days)
    except Exception:
        return val


TARGET_ORDER = [
    "date",
    "debit_account",
    "debit_name",
    "debit_inn",
    "credit_account",
    "credit_name",
    "credit_inn",
    "debit_amount",
    "credit_amount",
    "purpose",
]


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


ACC_PAT = re.compile(r"\b(\d{20})\b")
INN_PAT = re.compile(r"\b(\d{10}|\d{12})\b")

def extract_party(cell: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract (account, inn, name) from a compound cell that may look like:
    'Дебет\\n11111111111111111111\\n2222222222\\nООО \"Шляпа\"'
    Strategy:
      - Split into non-empty trimmed lines.
      - First 20-digit sequence => account.
      - First 10/12-digit sequence => inn.
      - Candidate name: last line with letters OR leftover text after removing numbers and keywords.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None, None, None
    s = str(cell)
    # Normalize separators
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Remove obvious labels
    s = re.sub(r"(?i)\b(дебет|кредит|инн|счет|счёт|р/с|р\.с\.)\b[:：]?", " ", s)
    lines = [ln.strip() for ln in re.split(r"[\n;|,]+", s) if ln and ln.strip()]
    text_joined = " ".join(lines)

    acc_m = ACC_PAT.search(text_joined)
    inn_m = INN_PAT.search(text_joined)

    account = acc_m.group(1) if acc_m else None
    inn = inn_m.group(1) if inn_m else None

    # heuristics for name: prefer last line with letters
    name = None
    for ln in reversed(lines):
        if re.search(r"[A-Za-zА-Яа-яЁё]", ln) and not re.fullmatch(r"\d+", ln):
            name = ln.strip().strip('"“”„«»').strip()
            break

    # Fallback: remove numbers and extra spaces from joined text
    if name is None:
        tmp = re.sub(r"\b\d+\b", " ", text_joined)
        tmp = re.sub(r"\s{2,}", " ", tmp).strip()
        if tmp:
            name = tmp

    return account, inn, name


def anonymize_to_fixed(df: pd.DataFrame, colmap: Dict[str, str], salt: str, date_shift_days: int) -> pd.DataFrame:
    work = pd.DataFrame()

    # If compound columns are provided, parse them to three fields
    if colmap.get("in_debit"):
        acc, inn, nm = zip(*df[colmap["in_debit"]].apply(extract_party))
        work["debit_account_raw"] = list(acc)
        work["debit_inn_raw"] = list(inn)
        work["debit_name_raw"] = list(nm)
    else:
        work["debit_account_raw"] = df[colmap["in_debit_acct"]] if colmap.get("in_debit_acct") else df.get("debit_account")
        work["debit_inn_raw"] = df[colmap["in_debit_inn"]] if colmap.get("in_debit_inn") else df.get("debit_inn")
        work["debit_name_raw"] = df[colmap["in_debit_name"]] if colmap.get("in_debit_name") else df.get("debit_name")

    if colmap.get("in_credit"):
        acc, inn, nm = zip(*df[colmap["in_credit"]].apply(extract_party))
        work["credit_account_raw"] = list(acc)
        work["credit_inn_raw"] = list(inn)
        work["credit_name_raw"] = list(nm)
    else:
        work["credit_account_raw"] = df[colmap["in_credit_acct"]] if colmap.get("in_credit_acct") else df.get("credit_account")
        work["credit_inn_raw"] = df[colmap["in_credit_inn"]] if colmap.get("in_credit_inn") else df.get("credit_inn")
        work["credit_name_raw"] = df[colmap["in_credit_name"]] if colmap.get("in_credit_name") else df.get("credit_name")

    # Required simple fields
    date_col = colmap.get("in_date") or ("date" if "date" in df.columns else None)
    purpose_col = colmap.get("in_purpose") or ("purpose" if "purpose" in df.columns else None)
    debit_amount_col = colmap.get("in_debit_amount") or ("debit_amount" if "debit_amount" in df.columns else None)
    credit_amount_col = colmap.get("in_credit_amount") or ("credit_amount" if "credit_amount" in df.columns else None)

    missing = [("date", date_col), ("purpose", purpose_col), ("debit_amount", debit_amount_col), ("credit_amount", credit_amount_col)]
    missing += [("debit_account_raw", work.get("debit_account_raw")),
                ("debit_inn_raw", work.get("debit_inn_raw")),
                ("debit_name_raw", work.get("debit_name_raw")),
                ("credit_account_raw", work.get("credit_account_raw")),
                ("credit_inn_raw", work.get("credit_inn_raw")),
                ("credit_name_raw", work.get("credit_name_raw"))]
    actually_missing = [k for k, v in missing if v is None]
    if actually_missing:
        raise ValueError(f"Не найдены необходимые столбцы/данные: {actually_missing}")

    work["date"] = df[date_col]
    work["purpose"] = df[purpose_col]
    work["debit_amount"] = df[debit_amount_col]
    work["credit_amount"] = df[credit_amount_col]

    # Hash PII columns from raw
    for raw, tgt, prefix in [
        ("debit_account_raw", "debit_account", "DEBIT_ACC_"),
        ("debit_name_raw", "debit_name", "DEBIT_NAME_"),
        ("debit_inn_raw", "debit_inn", "DEBIT_INN_"),
        ("credit_account_raw", "credit_account", "CREDIT_ACC_"),
        ("credit_name_raw", "credit_name", "CREDIT_NAME_"),
        ("credit_inn_raw", "credit_inn", "CREDIT_INN_"),
    ]:
        work[tgt] = work[raw].apply(lambda x: hmac_hex(salt, str(x), prefix=prefix, length=16) if x is not None else None)

    # Purpose redaction
    work["purpose"] = work["purpose"].apply(lambda x: redact_purpose(x, salt))

    # Date shift
    work["date"] = work["date"].apply(lambda x: shift_date(x, date_shift_days))

    # Coerce amounts to numeric (no hashing)
    work["debit_amount"] = to_numeric(work["debit_amount"])
    work["credit_amount"] = to_numeric(work["credit_amount"])

    # Final selection/reorder
    work = work[TARGET_ORDER]
    return work


def main():
    p = argparse.ArgumentParser(description="Anonymize Excel to fixed schema for ML training (supports compound debit/credit cells).")
    p.add_argument("--in", dest="input_path", required=True, help="Input Excel file path")
    p.add_argument("--out", dest="output_path", required=True, help="Output Excel file path")
    p.add_argument("--salt", dest="salt", required=True, help="Secret key for HMAC (keep it secret!)")
    p.add_argument("--sheet", dest="sheet_name", default=None, help="Sheet name (default: first)")
    # Either provide compound columns...
    p.add_argument("--in-debit", dest="in_debit", default=None, help="Column with compound debit cell (account+inn+name)")
    p.add_argument("--in-credit", dest="in_credit", default=None, help="Column with compound credit cell (account+inn+name)")
    # ...or separate columns:
    p.add_argument("--in-date", dest="in_date", default=None)
    p.add_argument("--in-debit-acct", dest="in_debit_acct", default=None)
    p.add_argument("--in-debit-name", dest="in_debit_name", default=None)
    p.add_argument("--in-debit-inn", dest="in_debit_inn", default=None)
    p.add_argument("--in-credit-acct", dest="in_credit_acct", default=None)
    p.add_argument("--in-credit-name", dest="in_credit_name", default=None)
    p.add_argument("--in-credit-inn", dest="in_credit_inn", default=None)
    p.add_argument("--in-debit-amount", dest="in_debit_amount", default=None)
    p.add_argument("--in-credit-amount", dest="in_credit_amount", default=None)
    p.add_argument("--in-purpose", dest="in_purpose", default=None)
    p.add_argument("--date-shift-days", dest="date_shift_days", type=int, default=0)
    args = p.parse_args()

    df = pd.read_excel(args.input_path, sheet_name=args.sheet_name)

    colmap = {
        "in_debit": args.in_debit,
        "in_credit": args.in_credit,
        "in_date": args.in_date,
        "in_debit_acct": args.in_debit_acct,
        "in_debit_name": args.in_debit_name,
        "in_debit_inn": args.in_debit_inn,
        "in_credit_acct": args.in_credit_acct,
        "in_credit_name": args.in_credit_name,
        "in_credit_inn": args.in_credit_inn,
        "in_debit_amount": args.in_debit_amount,
        "in_credit_amount": args.in_credit_amount,
        "in_purpose": args.in_purpose,
    }

    out_df = anonymize_to_fixed(df, colmap, args.salt, args.date_shift_days)

    # Write output
    out_df.to_excel(args.output_path, index=False)

if __name__ == "__main__":
    main()
