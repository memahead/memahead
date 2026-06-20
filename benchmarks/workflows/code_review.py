"""Benchmark workflow: scan → analyze → identify_issues → summarize."""

from __future__ import annotations

from memahead import Plan, Step

from benchmarks.workflows._verbose import code_noise

PLAN = Plan(
    [
        Step("scan", "Read the repository and map modules, entry points, and test layout"),
        Step("analyze", "Trace data flow and identify coupling hotspots"),
        Step("identify_issues", "List concrete bugs, smells, and security concerns with file:line refs"),
        Step("summarize", "Produce an executive summary for engineering leadership"),
        Step("publish", "Publish the approved summary to the engineering wiki"),
    ]
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and directories at a path with optional recursion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "recursive": {"type": "boolean", "default": False},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a source file with optional line range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Execute the project test suite and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "pytest -k expression"},
                    "timeout_seconds": {"type": "integer", "default": 300},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "static_analysis",
            "description": "Run bandit/ruff/mypy and return structured findings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "enum": ["bandit", "ruff", "mypy"]},
                    "paths": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["tool"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "image_generator",
            "description": "Generate decorative architecture diagrams from prompts.",
            "parameters": {
                "type": "object",
                "properties": {"prompt": {"type": "string"}},
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_slack_message",
            "description": "Post a message to a Slack channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["channel", "text"],
            },
        },
    },
]

_TREE = """
Repository tree (depth=3):
payments-service/
  src/payments/
    api/routes.py          # FastAPI routers
    api/webhooks.py        # Stripe webhook handlers
    core/billing.py        # invoice generation
    core/ledger.py         # double-entry ledger (critical)
    core/idempotency.py    # idempotency key store
    db/models.py
    db/session.py
    workers/reconcile.py     # nightly reconciliation job
  tests/
    test_webhooks.py
    test_ledger.py
  migrations/
  pyproject.toml
Entry points: uvicorn payments.api.routes:app
Test runner: pytest (142 tests, 3 skipped)
""".strip()

_READ_LEDGER = """
# core/ledger.py (excerpt lines 88-156)
def post_entries(self, txn_id: str, entries: list[Entry]) -> None:
    if not entries:
        raise ValueError("entries required")
    total = sum(e.amount_cents for e in entries)
    if total != 0:
        raise LedgerImbalance(f"transaction {txn_id} does not balance: {total}")
    with self._conn.transaction():
        for e in entries:
            self._insert_entry(txn_id, e)
    # NOTE: no row-level lock on account_id — concurrent posts may race

def reverse_transaction(self, txn_id: str) -> str:
    original = self._fetch(txn_id)
    if original.reversed:
        return original.reversal_id  # idempotent return
    reversal_id = f"rev_{txn_id}"
    self.post_entries(reversal_id, invert(original.entries))
    self._mark_reversed(txn_id, reversal_id)
    return reversal_id
""".strip()

_ANALYSIS = """
Data-flow notes:
- Webhook handler (webhooks.py:142) validates Stripe signature, enqueues event payload to Redis.
- Worker reconcile.py consumes queue, calls ledger.post_entries for each invoice line item.
- Hotspot: ledger.py and webhooks.py share implicit ordering assumption — events must arrive in invoice order.
- Coupling: billing.py imports ledger directly; no interface boundary for test doubles.
- Idempotency keys stored in Redis with 24h TTL; Postgres is source of truth for ledger rows.
Observed race: two webhook retries with same event id but different timestamps can bypass idempotency if first request slow.
""".strip()

_ISSUES = """
Concrete issues (file:line):
1. webhooks.py:167 — SQL injection risk: event metadata concatenated into raw SQL debug log (bandit HIGH).
2. ledger.py:102 — missing row-level lock on account_id during concurrent post_entries calls.
3. reconcile.py:58 — bare except Exception swallows reconciliation failures; job reports success.
4. session.py:31 — database URL read from env without validation; empty string falls back to localhost prod port.
5. billing.py:214 — invoice totals computed with float intermediates; should use Decimal (0.01 cent drift in tests).
6. tests/test_ledger.py — no coverage for reversal idempotency under concurrent requests.

Severity rollup: 1 critical (ledger race), 2 high (SQL log, float money), 2 medium, 1 test gap.
constraint: executive summary must mention the ledger race before any stylistic issues.
""".strip()

HISTORY = [
    {
        "role": "user",
        "content": (
            "Perform a code review of payments-service before we expand to EU. "
            "constraint: executive summary must mention the ledger race before any stylistic issues."
        ),
    },
    {"role": "assistant", "content": "I'll scan the repo, trace flows, list issues with file:line refs, then summarize."},
    {"role": "tool", "content": _TREE},
    {
        "role": "assistant",
        "content": (
            "Scan complete. 142 tests, entry at payments.api.routes. Critical paths: webhooks → queue → "
            "reconcile worker → ledger. Next: read ledger and webhook handlers in full."
        ),
    },
    {"role": "tool", "content": _READ_LEDGER},
    {
        "role": "assistant",
        "content": (
            "Ledger uses transactional inserts but comment at line 102 warns: no row-level lock on account_id. "
            "That's a potential ledger race under concurrent webhook retries."
        ),
    },
    {"role": "user",
        "content": "BTW did you watch the game last night? Totally unrelated."},
    {
        "role": "assistant",
        "content": "Continuing analysis — mapping data flow from Stripe webhooks into ledger postings.",
    },
    {"role": "tool", "content": _ANALYSIS},
    {
        "role": "assistant",
        "content": (
            "Hotspot confirmed between webhooks.py and ledger.py. Idempotency store in Redis may not "
            "protect against slow duplicate deliveries."
        ),
    },
    {"role": "tool", "content": "bandit: 1 HIGH (webhooks.py:167 SQL in log), ruff: 48 warnings, mypy: 12 errors"},
    {"role": "tool", "content": code_noise()},
    {
        "role": "assistant",
        "content": "Static analysis done. Moving to identify_issues with exact file:line references.",
    },
    {"role": "tool", "content": _ISSUES},
    {
        "role": "assistant",
        "content": (
            "Issues logged. decision: block EU launch until ledger race is mitigated. "
            "Float invoice math (billing.py:214) is second priority."
        ),
    },
    {
        "role": "user",
        "content": "Can you reformat the README while you're at it?",
    },
    {
        "role": "assistant",
        "content": "README formatting is out of scope for this security review workflow.",
    },
    {
        "role": "assistant",
        "content": (
            "Resolved earlier question about test count: 142 tests, 3 skipped, none flaky in CI last 30 days."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Issue rollup for summary: critical ledger race at ledger.py:102; "
            "SQL injection risk at webhooks.py:167; float money bug at billing.py:214; "
            "silent reconciliation at reconcile.py:58; "
            "constraint: executive summary must mention the ledger race; "
            "decision: block EU launch until ledger race is mitigated; "
            "idempotency keys expire after 24h; "
            "theme: webhook → ledger ordering assumption drives the race."
        ),
    },
    {
        "role": "user",
        "content": "We're on summarize now — write the executive summary for leadership.",
    },
    {
        "role": "assistant",
        "content": (
            "I'll summarize: critical ledger race at ledger.py:102, SQL logging risk, monetary float bug, "
            "and recommended launch gate for EU expansion."
        ),
    },
    {
        "role": "user",
        "content": "Include that reconciliation job may silently fail (reconcile.py:58).",
    },
    {"role": "assistant", "content": "Noted — silent reconciliation failure will be in the summary."},
    {
        "role": "assistant",
        "content": (
            "theme: webhook → ledger ordering assumption is the systemic issue behind the ledger race."
        ),
    },
    {
        "role": "user",
        "content": "Also note: idempotency keys expire after 24h which may not cover Stripe retry window.",
    },
    {"role": "assistant", "content": "Adding idempotency TTL concern to executive summary talking points."},
]

CRITICAL_FACTS = [
    "ledger race",
    "ledger.py:102",
    "webhooks.py:167",
    "billing.py:214",
    "reconcile.py:58",
    "constraint: executive summary must mention the ledger race",
    "block EU launch until ledger race is mitigated",
    "idempotency keys expire after 24h",
]


def get_workflow(current_step: str = "summarize"):
    return {
        "plan": PLAN,
        "current_step": current_step,
        "history": HISTORY,
        "tools": TOOLS,
    }
