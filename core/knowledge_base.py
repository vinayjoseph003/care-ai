# ================================================================
#  core/knowledge_base.py  — UNIFIED VERSION
#  ✅ [NEW] build_precaution_map() added
#  ✅ [NEW] build_all_chunks() now returns (chunks, sev_map, prec_map)
# ================================================================

import os
import re
import csv
import xml.etree.ElementTree as ET
import pandas as pd


def _get_col(df, keyword):
    for c in df.columns:
        if keyword.lower() in c.lower():
            return c
    return None


def _clean(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', str(text).strip())


def load_datasets(data_dir: str = "data") -> dict:
    paths = {
        "symptoms":    os.path.join(data_dir, "dataset.csv"),
        "description": os.path.join(data_dir, "symptom_Description.csv"),
        "precaution":  os.path.join(data_dir, "symptom_precaution.csv"),
        "severity":    os.path.join(data_dir, "Symptom-severity.csv"),
    }
    dfs = {}
    for key, path in paths.items():
        if not os.path.exists(path):
            print(f"⚠️  Warning: {path} not found. Skipping.")
            dfs[key] = pd.DataFrame()
        else:
            dfs[key] = pd.read_csv(path)
            dfs[key].columns = dfs[key].columns.str.strip()
            print(f"✅ Loaded {key}: {len(dfs[key])} rows")
    return dfs


def _extract_symptoms(row, cols) -> list:
    syms = []
    for col in cols:
        val = row.get(col, None)
        if pd.notna(val) and str(val).strip() not in ('', 'nan'):
            syms.append(str(val).strip().lower().replace('_', ' '))
    return syms


# ── [NEW] PRECAUTION MAP BUILDER ─────────────────────────────
def build_precaution_map(df_prec: pd.DataFrame) -> dict:
    """
    Build { disease_name: [precaution_1, precaution_2, ...] }
    from the symptom_precaution.csv DataFrame.
    """
    if df_prec.empty:
        return {}
    prec_dis_col = _get_col(df_prec, 'disease')
    if not prec_dis_col:
        return {}
    pcols  = [c for c in df_prec.columns if c != prec_dis_col]
    result = {}
    for _, row in df_prec.iterrows():
        disease = str(row[prec_dis_col]).strip()
        if not disease or disease == 'nan':
            continue
        precs = [
            str(row.get(col, '')).strip()
            for col in pcols
            if str(row.get(col, '')).strip() not in ('', 'nan', 'None')
        ]
        if precs:
            result[disease] = precs
    print(f"✅ Precaution map built: {len(result)} diseases")
    return result


def build_knowledge_chunks(dfs: dict) -> list:
    df_sym  = dfs.get("symptoms",    pd.DataFrame())
    df_desc = dfs.get("description", pd.DataFrame())
    df_prec = dfs.get("precaution",  pd.DataFrame())
    df_sev  = dfs.get("severity",    pd.DataFrame())

    sev_lookup = {}
    if not df_sev.empty:
        sym_col = _get_col(df_sev, 'symptom')
        wt_col  = _get_col(df_sev, 'weight')
        if sym_col and wt_col:
            for _, row in df_sev.iterrows():
                k = str(row[sym_col]).strip().lower().replace('_', ' ')
                try:    sev_lookup[k] = int(row[wt_col])
                except: pass

    sym_dis_col  = _get_col(df_sym,  'disease')
    desc_dis_col = _get_col(df_desc, 'disease')
    prec_dis_col = _get_col(df_prec, 'disease')
    sym_cols     = [c for c in df_sym.columns if c != sym_dis_col] \
                   if not df_sym.empty else []

    all_diseases = set()
    if not df_sym.empty  and sym_dis_col:
        all_diseases.update(df_sym[sym_dis_col].dropna().str.strip().unique())
    if not df_desc.empty and desc_dis_col:
        all_diseases.update(df_desc[desc_dis_col].dropna().str.strip().unique())

    desc_lookup = {}
    if not df_desc.empty and desc_dis_col:
        dc = next((c for c in df_desc.columns
                   if c != desc_dis_col and 'desc' in c.lower()), None)
        if not dc:
            others = [c for c in df_desc.columns if c != desc_dis_col]
            dc = others[0] if others else None
        if dc:
            for _, row in df_desc.iterrows():
                desc_lookup[str(row[desc_dis_col]).strip()] = \
                    str(row[dc]).strip()

    prec_lookup = {}
    if not df_prec.empty and prec_dis_col:
        pcols = [c for c in df_prec.columns if c != prec_dis_col]
        for _, row in df_prec.iterrows():
            k, precs = str(row[prec_dis_col]).strip(), []
            for col in pcols:
                val = row.get(col, None)
                if pd.notna(val) and str(val).strip() not in ('', 'nan'):
                    precs.append(str(val).strip())
            if precs:
                prec_lookup[k] = precs

    chunks = []
    for disease in sorted(all_diseases):
        if not disease:
            continue

        symptoms = []
        if not df_sym.empty and sym_dis_col:
            rows = df_sym[df_sym[sym_dis_col].str.strip() == disease]
            for _, row in rows.iterrows():
                symptoms.extend(_extract_symptoms(row, sym_cols))
            symptoms = list(dict.fromkeys(symptoms))

        description = desc_lookup.get(disease, "")
        precautions = prec_lookup.get(disease, [])
        sym_scores  = {s: sev_lookup.get(s, 0) for s in symptoms}
        max_sev     = max(sym_scores.values()) if sym_scores else 0
        total_sev   = sum(sym_scores.values())
        sev_label   = (
            "HIGH"   if max_sev >= 5 or total_sev >= 13 else
            "MEDIUM" if max_sev >= 3 or total_sev >= 7  else
            "LOW"
        )

        text_parts = [
            f"Disease: {disease}.",
            f"{disease} is a medical condition.",
        ]
        if description:
            text_parts.append(description)
        if symptoms:
            text_parts.append(
                f"Symptoms of {disease} include: {', '.join(symptoms)}.")
            text_parts.append(
                f"A patient with {disease} may experience: "
                f"{', '.join(symptoms[:8])}.")
        if precautions:
            text_parts.append(
                f"Precautions for {disease}: {'; '.join(precautions)}.")
        text_parts.append(f"Severity level: {sev_label}.")

        chunks.append({
            "id":               disease.lower().replace(' ', '_'),
            "disease":          disease,
            "category":         "Disease-Symptom",
            "source":           "Mendeley Medical Dataset",
            "symptoms":         symptoms,
            "precautions":      precautions,
            "severity":         sev_label,
            "max_severity_score": max_sev,
            "symptom_scores":   sym_scores,
            "text":             " ".join(text_parts),
        })

    print(f"✅ CSV chunks built: {len(chunks)} diseases")
    return chunks


def get_symptom_severity_map(dfs: dict) -> dict:
    df_sev = dfs.get("severity", pd.DataFrame())
    if df_sev.empty:
        return {}
    sym_col = _get_col(df_sev, 'symptom')
    wt_col  = _get_col(df_sev, 'weight')
    if not sym_col or not wt_col:
        return {}
    result = {}
    for _, row in df_sev.iterrows():
        k = str(row[sym_col]).strip().lower().replace('_', ' ')
        try:    result[k] = int(row[wt_col])
        except: pass
    return result


MEDQUAD_CATEGORIES = {
    "CancerQA":                               "Oncology",
    "Diabetes_and_Digestive_and_Kidney":      "Endocrinology / Gastroenterology",
    "Disease_Control_and_Prevention":         "General Medicine / CDC",
    "Genetic_and_Rare_Diseases":              "Genetics / Rare Diseases",
    "growth_hormone_receptor":                "Endocrinology",
    "Heart_Lung_and_Blood":                   "Cardiology / Pulmonology",
    "MedicalQuestionAnswering":               "General Medicine",
    "Neurological_Disorders_and_Stroke":      "Neurology",
    "OtherQA":                                "General Medicine",
    "SeniorHealth":                           "Geriatrics / Senior Health",
}


def _get_medquad_category(filename: str) -> str:
    for key, cat in MEDQUAD_CATEGORIES.items():
        if key.lower() in filename.lower():
            return cat
    return "General Medicine"


def parse_medquad(data_dir: str = "data") -> list:
    medquad_dir = None
    for item in os.listdir(data_dir):
        full = os.path.join(data_dir, item)
        if os.path.isdir(full) and 'medquad' in item.lower():
            medquad_dir = full
            break

    if not medquad_dir:
        print("⚠️  MedQuAD folder not found in data/. Skipping.")
        return []

    chunks = []
    total_files = 0

    for fname in os.listdir(medquad_dir):
        if not fname.endswith('.csv'):
            continue

        fpath    = os.path.join(medquad_dir, fname)
        category = _get_medquad_category(fname)
        total_files += 1

        try:
            df = pd.read_csv(fpath, on_bad_lines='skip')
            df.columns = df.columns.str.strip()

            q_col = next((c for c in df.columns
                          if 'question' in c.lower() or c.lower() == 'q'), None)
            a_col = next((c for c in df.columns
                          if 'answer' in c.lower() or c.lower() == 'a'), None)

            if not q_col and len(df.columns) >= 1:
                q_col = df.columns[0]
            if not a_col and len(df.columns) >= 2:
                a_col = df.columns[1]

            if not q_col or not a_col:
                print(f"  ⚠️  {fname}: Could not detect Q/A columns. Skipping.")
                continue

            file_chunks = 0
            for _, row in df.iterrows():
                question = _clean(row.get(q_col, ''))
                answer   = _clean(row.get(a_col, ''))

                if not question or not answer or len(answer) < 20:
                    continue

                text = (
                    f"Question: {question} "
                    f"Answer: {answer} "
                    f"Medical Q&A about: {question}"
                )

                chunks.append({
                    "id":       f"medquad_{len(chunks)}",
                    "category": category,
                    "source":   f"MedQuAD — {fname}",
                    "question": question,
                    "text":     text,
                })
                file_chunks += 1

            print(f"  ✅ {fname}: {file_chunks} Q&A chunks")

        except Exception as e:
            print(f"  ⚠️  {fname}: Error — {e}")

    print(f"✅ MedQuAD parsed: {len(chunks)} Q&A chunks from {total_files} files")
    return chunks


def parse_medlineplus(data_dir: str = "data") -> list:
    xml_file = None
    for fname in os.listdir(data_dir):
        if fname.startswith('mplus') and fname.endswith('.xml'):
            xml_file = os.path.join(data_dir, fname)
            break

    if not xml_file:
        print("⚠️  MedlinePlus XML not found in data/. Skipping.")
        return []

    print(f"📄 Parsing MedlinePlus XML: {os.path.basename(xml_file)}")

    chunks  = []
    skipped = 0

    try:
        tree   = ET.parse(xml_file)
        root   = tree.getroot()
        topics = root.findall('.//health-topic')
        if not topics:
            topics = root.findall('health-topic')

        for topic in topics:
            title      = _clean(topic.get('title', ''))
            url        = topic.get('url', '')
            summary_el = topic.find('full-summary')
            summary    = ""
            if summary_el is not None and summary_el.text:
                summary = re.sub(r'<[^>]+>', ' ', summary_el.text)
                summary = _clean(summary)

            also_called = [_clean(ac.text) for ac in topic.findall('also-called') if ac.text]
            groups      = [_clean(g.text)  for g  in topic.findall('group')       if g.text]
            related     = [_clean(rt.text) for rt in topic.findall('related-topic') if rt.text]

            if not title or not summary:
                skipped += 1
                continue

            text_parts = [f"Health Topic: {title}."]
            if also_called:
                text_parts.append(f"{title} is also known as: {', '.join(also_called)}.")
            if groups:
                text_parts.append(f"Medical category: {', '.join(groups)}.")
            text_parts.append(summary[:800])
            if related:
                text_parts.append(f"Related topics: {', '.join(related[:5])}.")

            chunks.append({
                "id":       f"mplus_{len(chunks)}",
                "title":    title,
                "category": groups[0] if groups else "General Medicine",
                "source":   "MedlinePlus (NIH)",
                "url":      url,
                "text":     " ".join(text_parts),
            })

        print(f"✅ MedlinePlus parsed: {len(chunks)} health topic chunks "
              f"({skipped} skipped — no summary)")

    except ET.ParseError as e:
        print(f"❌ MedlinePlus XML parse error: {e}")
    except Exception as e:
        print(f"❌ MedlinePlus error: {e}")

    return chunks


def parse_icd10(data_dir: str = "data") -> list:
    icd_file = os.path.join(data_dir, "icd10cm_order_2026.txt")

    if not os.path.exists(icd_file):
        print("⚠️  icd10cm_order_2026.txt not found in data/. Skipping.")
        return []

    print(f"📄 Parsing ICD-10 file...")

    ICD_CATEGORIES = {
        'A': 'Infectious Diseases', 'B': 'Infectious Diseases',
        'C': 'Oncology',            'D': 'Blood / Oncology',
        'E': 'Endocrinology',       'F': 'Psychiatry / Mental Health',
        'G': 'Neurology',           'H': 'Eye / Ear Disorders',
        'I': 'Cardiology',          'J': 'Respiratory',
        'K': 'Gastroenterology',    'L': 'Dermatology',
        'M': 'Orthopedics',         'N': 'Urology / Nephrology',
        'O': 'Obstetrics',          'P': 'Pediatrics',
        'Q': 'Congenital',          'R': 'Symptoms / Signs',
        'S': 'Injury / Trauma',     'T': 'Poisoning / Toxicology',
        'Z': 'Preventive / Wellness',
    }

    chunks        = []
    prefix_groups = {}

    try:
        with open(icd_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if len(line) < 77:
                    continue
                if line[14:15].strip() == '1':
                    continue

                code      = line[6:14].strip()
                short_desc = _clean(line[16:77])
                long_desc  = _clean(line[77:]) if len(line) > 77 else short_desc

                if not code or not long_desc:
                    continue

                prefix   = code[:3]
                category = ICD_CATEGORIES.get(code[0], 'General Medicine')

                if prefix not in prefix_groups:
                    prefix_groups[prefix] = {
                        "prefix":   prefix,
                        "category": category,
                        "codes":    [],
                    }
                prefix_groups[prefix]["codes"].append({
                    "code":       code,
                    "short_desc": short_desc,
                    "long_desc":  long_desc,
                })

        for prefix, group in prefix_groups.items():
            codes    = group["codes"]
            category = group["category"]
            title    = codes[0]["short_desc"] if codes else prefix
            descriptions = [
                f"{c['code']}: {c['long_desc']}"
                for c in codes[:20]
            ]
            text = (
                f"ICD-10 codes starting with {prefix}. "
                f"Medical category: {category}. "
                f"Conditions: {' | '.join(descriptions)}"
            )
            chunks.append({
                "id":         f"icd10_{prefix}",
                "prefix":     prefix,
                "title":      title,
                "category":   category,
                "source":     "ICD-10-CM 2026 (CMS)",
                "code_count": len(codes),
                "text":       text,
            })

        print(f"✅ ICD-10 parsed: {len(chunks)} prefix-grouped chunks")

    except Exception as e:
        print(f"❌ ICD-10 parse error: {e}")

    return chunks


# ── UNIFIED LOADER ────────────────────────────────────────────
def build_all_chunks(data_dir: str = "data") -> tuple:
    """
    Returns (all_chunks, severity_map, precaution_map)
    ✅ Now returns precaution_map as the third element.
    """
    print("\n" + "="*55)
    print("  📚 Loading All Knowledge Sources")
    print("="*55)

    all_chunks = []

    print("\n[1/4] Loading disease-symptom CSVs...")
    dfs        = load_datasets(data_dir)
    csv_chunks = build_knowledge_chunks(dfs)
    sev_map    = get_symptom_severity_map(dfs)
    prec_map   = build_precaution_map(dfs.get("precaution", pd.DataFrame()))  # [NEW]
    all_chunks.extend(csv_chunks)

    print("\n[2/4] Parsing MedQuAD Q&A files...")
    mq_chunks  = parse_medquad(data_dir)
    all_chunks.extend(mq_chunks)

    print("\n[3/4] Parsing MedlinePlus XML...")
    mp_chunks  = parse_medlineplus(data_dir)
    all_chunks.extend(mp_chunks)

    print("\n[4/4] Parsing ICD-10 codes...")
    icd_chunks = parse_icd10(data_dir)
    all_chunks.extend(icd_chunks)

    print("\n" + "="*55)
    print(f"  ✅ TOTAL KNOWLEDGE BASE: {len(all_chunks)} chunks")
    print(f"     • Disease-Symptom CSV : {len(csv_chunks)}")
    print(f"     • MedQuAD Q&A         : {len(mq_chunks)}")
    print(f"     • MedlinePlus Topics  : {len(mp_chunks)}")
    print(f"     • ICD-10 Groups       : {len(icd_chunks)}")
    print(f"     • Severity map        : {len(sev_map)} symptoms")
    print(f"     • Precaution map      : {len(prec_map)} diseases")
    print("="*55 + "\n")

    return all_chunks, sev_map, prec_map   # ← [NEW] third return value