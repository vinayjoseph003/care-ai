# ================================================================
#  utils/display.py — Clinical display
#  ✅ print_triage_header() — fires before streaming
#  ✅ [NEW] display_treatment() — home care, escalation, lifestyle
#  ✅ [NEW] display_red_flags() — warning symptoms panel
#  ✅ [NEW] display_mental_health() — compassionate check-in
# ================================================================

TRIAGE_CONFIG = {
    "SELF_CARE":   {"icon": "🟢", "label": "SELF-CARE",    "bar": "▓▓░░"},
    "SEE_DOCTOR":  {"icon": "🟡", "label": "SEE A DOCTOR", "bar": "▓▓▓░"},
    "URGENT_CARE": {"icon": "🔴", "label": "URGENT CARE",  "bar": "▓▓▓▓"},
    "EMERGENCY":   {"icon": "🚨", "label": "EMERGENCY",    "bar": "████"},
}

PROB_ICON = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}

SPECIALIST_ICON_MAP = {
    "gastro": "🫁", "cardio": "❤️",  "neuro": "🧠",
    "derma":  "🩺", "psych":  "🧘",  "pulmo": "🫁",
    "ortho":  "🦴", "urolo":  "💧",  "gynae": "🌸",
    "gynec":  "🌸", "endo":   "⚗️",  "general": "👨‍⚕️",
    "emergency": "🚨",
}


def confidence_bar(score: int) -> str:
    filled = round(score / 10)
    bar    = '█' * filled + '░' * (10 - filled)
    color  = "🟢" if score >= 80 else "🟡" if score >= 55 else "🔴"
    return f"[{bar}] {color}"


def _specialist_icon(spec: str) -> str:
    spec_lower = spec.lower()
    return next(
        (v for k, v in SPECIALIST_ICON_MAP.items() if k in spec_lower), "🩺"
    )


def print_triage_header(triage_level: str, diff_dx: list, specialist: str = ""):
    """Fires BEFORE streaming — user sees structure while tokens arrive."""
    t_cfg = TRIAGE_CONFIG.get(triage_level, TRIAGE_CONFIG["SEE_DOCTOR"])

    print()
    print("═" * 64)
    print("  🏥 Care-AI — Clinical Assessment")
    print("═" * 64)
    print(f"\n  {t_cfg['icon']} TRIAGE: {t_cfg['label']}  {t_cfg['bar']}")

    if diff_dx:
        print("\n  📋 Differential Diagnosis:")
        for d in diff_dx[:3]:
            icon = PROB_ICON.get(d.get('probability', 'LOW'), '⚪')
            rank = d.get('rank', '?')
            cond = d.get('condition', 'Unknown')
            prob = d.get('probability', '')
            print(f"    {rank}. {icon} {cond} — {prob} probability")

    if specialist:
        s_icon = _specialist_icon(specialist)
        print(f"\n  {s_icon} Recommended Specialist: {specialist}")

    print("\n" + "─" * 64)


# ── [NEW] TREATMENT GUIDANCE DISPLAY ─────────────────────────

def display_treatment(treatment: dict, triage: str):
    """
    Display ethical treatment guidance — no drug names.
    Layout adapts to triage level:
      SELF_CARE   → full home care section
      SEE_DOCTOR  → home care + escalation
      URGENT_CARE / EMERGENCY → escalation first, brief home care
    """
    if not treatment:
        return

    print("\n" + "─" * 64)
    print("  🩹 Treatment Guidance")
    print("─" * 64)

    home_care  = treatment.get('home_care', [])
    escalation = treatment.get('escalation_timeline', '')
    lifestyle  = treatment.get('lifestyle_tips', [])
    avoid      = treatment.get('what_to_avoid', [])

    # For urgent/emergency: escalation comes first
    if triage in ("URGENT_CARE", "EMERGENCY"):
        if escalation:
            print(f"\n  ⚡ Next Steps:")
            print(f"    {escalation}")
        if home_care:
            print(f"\n  🏠 While you wait / home support:")
            for step in home_care[:3]:   # keep brief for urgent cases
                print(f"    • {step}")

    else:
        # Self-care / see doctor: home care first
        if home_care:
            print(f"\n  🏠 Home Care Steps:")
            for step in home_care:
                print(f"    • {step}")

        if escalation:
            print(f"\n  ⏰ When to Escalate:")
            print(f"    {escalation}")

    # Lifestyle tips always shown
    if lifestyle:
        print(f"\n  🌱 Lifestyle & Prevention:")
        for tip in lifestyle:
            print(f"    • {tip}")

    # What to avoid
    if avoid:
        print(f"\n  🚫 What to Avoid:")
        for item in avoid:
            print(f"    • {item}")


# ── [NEW] RED FLAG ALERTS DISPLAY ────────────────────────────

def display_red_flags(red_flags: dict):
    """Display condition-specific warning symptoms to watch for."""
    if not red_flags:
        return

    flags     = red_flags.get('red_flags', [])
    threshold = red_flags.get('emergency_threshold', '')

    if not flags:
        return

    print("\n" + "─" * 64)
    print("  🚨 Red Flag Symptoms — Seek Emergency Care If You Notice:")
    print("─" * 64)
    for flag in flags:
        print(f"    ⚠️  {flag}")
    if threshold:
        print(f"\n  → {threshold}")


# ── [NEW] MENTAL HEALTH CHECK-IN DISPLAY ─────────────────────

def display_mental_health(message: str):
    """Display a compassionate mental health check-in."""
    if not message:
        return
    print("\n" + "─" * 64)
    print("  💙 A Note on Your Wellbeing")
    print("─" * 64)
    print(f"  {message}")
    print("  (If you're feeling overwhelmed, speaking to someone you trust")
    print("   or a mental health professional can really help.)")


# ── MAIN RESPONSE DISPLAY ─────────────────────────────────────

def display_response(result: dict):
    rtype = result.get('type')

    if rtype == 'emergency':
        print("\n" + "🚨" * 20)
        print(result['message'])
        print("🚨" * 20 + "\n")

    elif rtype == 'followup':
        count = result.get('followup_count', 1)
        sdim  = result.get('socrates_dim', '')
        label = f" [{sdim}]" if sdim else ""
        print(f"\n🤝 Care-AI [{count}/3]{label}:")
        print(f"   {result['message']}\n")

    elif rtype == 'followup_check':
        print()  # already printed in careai.py

    elif rtype == 'answer':
        score   = result.get('confidence', 0)
        risk    = result.get('hallucination_risk', 'N/A')
        verdict = result.get('fact_check', 'N/A')
        lang    = result.get('language', 'N/A')
        sev     = result.get('severity', 'N/A')
        triage  = result.get('triage', 'SEE_DOCTOR')
        sources = result.get('sources', [])

        risk_icon    = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(risk, "⚪")
        sev_icon     = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "EMERGENCY": "🚨"}.get(sev, "⚪")
        verdict_icon = {"VERIFIED": "✅", "PARTIALLY_VERIFIED": "🟡", "UNVERIFIED": "🔴"}.get(verdict, "⚪")

        # Header + answer text + XAI already streamed live by careai.py
        # display_response() handles the footer sections

        # [NEW] Treatment guidance
        display_treatment(result.get('treatment', {}), triage)

        # [NEW] Red flags
        display_red_flags(result.get('red_flags', {}))

        # [NEW] Mental health check-in
        display_mental_health(result.get('mental_health_msg', ''))

        # Sources
        if sources:
            print("\n" + "─" * 64)
            print("  📚 Sources Referenced:")
            for s in sources:
                print(f"    {s}")

        # Metrics footer
        print("\n" + "─" * 64)
        print(f"  📊 Confidence      : {score}% {confidence_bar(score)}")
        print(f"  {risk_icon} Hallucination Risk : {risk}")
        print(f"  {verdict_icon} Fact-check        : {verdict}")
        print(f"  {sev_icon} Severity           : {sev}")
        print(f"  🌐 Language         : {lang}")
        print("═" * 64 + "\n")


def display_symptom_score(score_result: dict):
    level = score_result.get('severity_level', 'N/A')
    icon  = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(level, "⚪")

    print("\n" + "─" * 44)
    print("  🩺 Symptom Severity Assessment")
    print("─" * 44)
    for sym, wt in score_result.get('symptom_scores', {}).items():
        bar = "█" * wt + "░" * (7 - min(wt, 7))
        print(f"  {sym:<30} [{bar}] {wt}/7")
    print(f"\n  Total Score : {score_result.get('total_score', 0)}")
    print(f"  {icon} Severity    : {level}")
    if score_result.get('high_risk_symptoms'):
        print(f"  ⚠️  High-risk  : {', '.join(score_result['high_risk_symptoms'])}")
    print(f"\n  {score_result.get('recommendation','')}")
    print("─" * 44 + "\n")