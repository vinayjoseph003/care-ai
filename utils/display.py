# ================================================================
#  utils/display.py — Clinical display with triage + differential
# ================================================================

TRIAGE_CONFIG = {
    "SELF_CARE":   {"icon": "🟢", "label": "SELF-CARE",    "bar": "▓▓░░"},
    "SEE_DOCTOR":  {"icon": "🟡", "label": "SEE A DOCTOR", "bar": "▓▓▓░"},
    "URGENT_CARE": {"icon": "🔴", "label": "URGENT CARE",  "bar": "▓▓▓▓"},
    "EMERGENCY":   {"icon": "🚨", "label": "EMERGENCY",    "bar": "████"},
}

PROB_ICON = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}


def confidence_bar(score: int) -> str:
    filled = round(score / 10)
    bar    = '█' * filled + '░' * (10 - filled)
    color  = "🟢" if score >= 80 else "🟡" if score >= 55 else "🔴"
    return f"[{bar}] {color}"


def display_response(result: dict):
    rtype = result.get('type')

    if rtype == 'emergency':
        print("\n" + "🚨" * 20)
        print(result['message'])
        print("🚨" * 20 + "\n")

    elif rtype == 'followup':
        count  = result.get('followup_count', 1)
        sdim   = result.get('socrates_dim', '')
        label  = f" [{sdim}]" if sdim else ""
        print(f"\n🤝 Care-AI [{count}/3]{label}:")
        print(f"   {result['message']}\n")

    elif rtype == 'answer':
        score    = result.get('confidence', 0)
        risk     = result.get('hallucination_risk', 'N/A')
        verdict  = result.get('fact_check', 'N/A')
        lang     = result.get('language', 'N/A')
        sev      = result.get('severity', 'N/A')
        triage   = result.get('triage', 'SEE_DOCTOR')
        spec     = result.get('specialist', '')
        diff     = result.get('differential', [])
        sources  = result.get('sources', [])

        t_cfg        = TRIAGE_CONFIG.get(triage, TRIAGE_CONFIG['SEE_DOCTOR'])
        risk_icon    = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴"}.get(risk, "⚪")
        sev_icon     = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴","EMERGENCY":"🚨"}.get(sev,"⚪")
        verdict_icon = {"VERIFIED":"✅","PARTIALLY_VERIFIED":"🟡","UNVERIFIED":"🔴"}.get(verdict,"⚪")

        print("\n" + "═" * 64)
        print("  🏥 Care-AI — Clinical Assessment")
        print("═" * 64)

        # ── Triage banner (most important, shown first) ──
        print(f"\n  {t_cfg['icon']} TRIAGE: {t_cfg['label']}  {t_cfg['bar']}")

        # ── Differential diagnosis ──
        if diff:
            print("\n  📋 Differential Diagnosis:")
            for d in diff:
                icon = PROB_ICON.get(d.get('probability','LOW'), '⚪')
                print(f"    {d['rank']}. {icon} {d['condition']} "
                      f"— {d.get('probability','')} probability")

        # ── Specialist referral ──
        if spec:
            spec_lower = spec.lower()
            spec_icon  = next(
                (v for k, v in {
                    "gastro":"🫁","cardio":"❤️","neuro":"🧠",
                    "derma":"🩺","psych":"🧘","pulmo":"🫁",
                    "ortho":"🦴","urolo":"💧","gynae":"🌸",
                    "gynec":"🌸","endo":"⚗️","general":"👨‍⚕️",
                    "emergency":"🚨"
                }.items() if k in spec_lower), "🩺"
            )
            print(f"\n  {spec_icon} Recommended Specialist: {spec}")

        print("\n" + "─" * 64)

        # ── Main answer (with inline citations) ──
        print(f"\n{result['message']}")

        # ── Sources ──
        if sources:
            print("\n" + "─" * 64)
            print("  📚 Sources Referenced:")
            for s in sources:
                print(f"    {s}")

        # ── Metrics ──
        print("\n" + "─" * 64)
        print(f"  📊 Confidence      : {score}% {confidence_bar(score)}")
        print(f"  {risk_icon} Hallucination Risk : {risk}")
        print(f"  {verdict_icon} Fact-check        : {verdict}")
        print(f"  {sev_icon} Severity           : {sev}")
        print(f"  🌐 Language         : {lang}")

        # ── XAI plain explanation ──
        if result.get('xai_explanation'):
            print("\n" + "─" * 64)
            print(result['xai_explanation'])

        print("═" * 64 + "\n")


def display_symptom_score(score_result: dict):
    level = score_result.get('severity_level', 'N/A')
    icon  = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴"}.get(level,"⚪")

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