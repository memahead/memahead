"""Benchmark workflow: ingest → clean → explore → model → validate → interpret."""

from __future__ import annotations

from memahead import Plan, Step

from benchmarks.workflows._verbose import data_noise

PLAN = Plan(
    [
        Step("ingest", "Load raw CSV exports and validate schema against the data dictionary"),
        Step("clean", "Fix nulls, deduplicate sessions, and normalize timestamps to UTC"),
        Step("explore", "Compute descriptive statistics and segment-level breakdowns"),
        Step("model", "Fit retention model and compute confidence intervals"),
        Step("validate", "Run holdout validation and sensitivity analysis"),
        Step("interpret", "Explain findings for the product leadership review"),
        Step("present", "Deliver the leadership deck with narrative and charts"),
    ]
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "load_dataset",
            "description": "Load a CSV or Parquet file into a dataframe with schema inference.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute SQL against the connected warehouse (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 1000},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_chart",
            "description": "Generate matplotlib charts and return base64 PNG.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {"type": "string", "enum": ["line", "bar", "hist"]},
                    "spec": {"type": "object"},
                },
                "required": ["chart_type", "spec"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fit_model",
            "description": "Fit sklearn/statsmodels model with reproducible seed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "features": {"type": "array", "items": {"type": "string"}},
                    "target": {"type": "string"},
                    "seed": {"type": "integer", "default": 42},
                },
                "required": ["model", "features", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Email analysis artifacts to stakeholders.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "array", "items": {"type": "string"}},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_lookup",
            "description": "Look up public holidays for scheduling meetings.",
            "parameters": {
                "type": "object",
                "properties": {"region": {"type": "string"}},
                "required": ["region"],
            },
        },
    },
]

_INGEST = """
Loaded: exports/sessions_2025Q1.csv (38,412,902 rows, 41 columns)
Schema validation against data dictionary v3.2:
- session_id: string OK
- user_id: string OK
- started_at: 2,301 rows failed ISO8601 parse (ingest step quarantined)
- country: ISO-3166 OK
- plan_tier: enum OK
- mrr_cents: integer OK
constraint: all revenue numbers must be reported in USD excluding VAT
Warning: duplicate session_id count 14,882 (0.039%) — forwarded to clean step
File hash: sha256:9f2c...a11
""".strip()

_CLEAN = """
Clean step results:
- Deduplicated 14,882 session rows keeping earliest started_at
- Dropped 2,301 rows with unparseable timestamps
- Normalized all timestamps to UTC
- Imputed country missing (0.7%) with 'UNKNOWN' bucket
- decision: exclude enterprise SSO pilot users (flag sso_pilot=true, n=4,201)
Final row count: 38,395,518 sessions
""".strip()

_EXPLORE = """
Exploratory statistics (post-clean):
Overall 30-day retention: 42.3%
Segment breakdown:
- US: 44.1% retention, 61% of sessions
- EU: 39.8% retention, 22% of sessions — EU GDPR consent banner cohort underperforms by 3.2pp
- APAC: 41.0% retention
Plan tier:
- free: 28.4%
- pro: 51.2%
- team: 58.7%
Observation: retention cliff at day 7 for free users (-11pp vs day 6)
Note: chart generation for leadership deck deferred until interpret step
""".strip()

_MODEL = """
Model: logistic regression + cluster-robust SE (user_id)
Target: retained_day_30
Features: plan_tier, region, onboarding_checklist_complete, first_week_active_days
Holdout: 20% user-level split, seed=42
Results:
- EU GDPR consent banner cohort underperforms by 3.2pp (p=0.003)
- onboarding_checklist_complete associated with +6.8pp retention (p<0.001)
- key finding: Q1 cohort retention 42.3% overall after SSO pilot exclusion
Confidence intervals reported at 95%.
""".strip()

_VALIDATE = """
Validation / sensitivity:
- Re-ran without EU segment: coefficients stable within 0.4pp
- Alternative model (Cox PH): same ranking of features
- Sensitivity: including SSO pilot users shifts EU gap to 2.1pp — still significant
- unresolved: whether banner UX change in March caused consent drop or seasonality
Holdout AUC: 0.71
""".strip()

HISTORY = [
    {
        "role": "user",
        "content": (
            "Analyze Q1 session retention for the board deck. "
            "constraint: all revenue numbers must be reported in USD excluding VAT."
        ),
    },
    {"role": "assistant", "content": "Pipeline: ingest → clean → explore → model → validate → interpret."},
    {"role": "tool", "content": _INGEST},
    {"role": "assistant", "content": "Ingest done. Duplicate sessions and bad timestamps flagged for cleaning."},
    {"role": "tool", "content": _CLEAN},
    {
        "role": "assistant",
        "content": (
            "Clean complete. decision: exclude enterprise SSO pilot users from retention metrics."
        ),
    },
    {"role": "user", "content": "What should we order for the team lunch on Friday?"},
    {"role": "assistant", "content": "Continuing exploration — computing segment retention tables."},
    {"role": "tool", "content": _EXPLORE},
    {"role": "tool", "content": data_noise()},
    {
        "role": "assistant",
        "content": (
            "Exploration shows EU GDPR consent banner cohort underperforms by 3.2pp. "
            "Free users have a day-7 retention cliff."
        ),
    },
    {"role": "tool", "content": _MODEL},
    {
        "role": "assistant",
        "content": (
            "Model confirms EU effect and onboarding checklist lift. "
            "key finding: Q1 cohort retention 42.3% overall after SSO pilot exclusion."
        ),
    },
    {
        "role": "user",
        "content": "Can you also forecast Q4 revenue tonight?",
    },
    {
        "role": "assistant",
        "content": "Revenue forecasting is out of scope — this workflow ends at interpretation of Q1 retention.",
    },
    {"role": "tool", "content": _VALIDATE},
    {"role": "assistant", "content": "Validation stable. Ready for interpret step for product leadership."},
    {
        "role": "assistant",
        "content": (
            "Interpretation rollup for leadership deck: "
            "key finding: Q1 cohort retention 42.3% overall after SSO pilot exclusion; "
            "EU GDPR consent banner cohort underperforms by 3.2pp; "
            "constraint: all revenue numbers must be reported in USD excluding VAT; "
            "decision: exclude enterprise SSO pilot users; "
            "onboarding_checklist_complete associated with +6.8pp retention; "
            "day-7 retention cliff for free tier; "
            "Confidence intervals reported at 95%."
        ),
    },
    {
        "role": "user",
        "content": "We're at interpret — explain findings for leadership. Focus on actionable EU and onboarding insights.",
    },
    {
        "role": "assistant",
        "content": (
            "I'll interpret: EU consent UX, onboarding checklist impact, and overall Q1 retention baseline."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Resolved: duplicate session handling already complete — no need to revisit ingest logic."
        ),
    },
    {
        "role": "user",
        "content": "Remind leadership that VAT exclusion applies to any revenue slide.",
    },
    {"role": "assistant", "content": "Will state USD excluding VAT on every revenue figure in the narrative."},
    {
        "role": "assistant",
        "content": (
            "theme: day-7 retention cliff for free tier is the primary onboarding lever for Q2 experiments."
        ),
    },
    {
        "role": "user",
        "content": "Include confidence intervals at 95% when citing model effects.",
    },
    {"role": "assistant", "content": "Confidence intervals at 95% will be cited for all modeled effects."},
]

CRITICAL_FACTS = [
    "key finding: Q1 cohort retention 42.3% overall",
    "EU GDPR consent banner cohort underperforms by 3.2pp",
    "constraint: all revenue numbers must be reported in USD excluding VAT",
    "decision: exclude enterprise SSO pilot users",
    "onboarding_checklist_complete associated with +6.8pp retention",
    "day-7 retention cliff for free tier",
    "Confidence intervals reported at 95%",
]


def get_workflow(current_step: str = "interpret"):
    return {
        "plan": PLAN,
        "current_step": current_step,
        "history": HISTORY,
        "tools": TOOLS,
    }
