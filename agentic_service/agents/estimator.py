"""
CorpBrain — Story Point Estimator
===================================
Rule-based Fibonacci estimation from (task_type, priority).
No external API needed.

When your own estimator model is trained on Kaggle, plug it in here.
"""

FIBONACCI = {1, 2, 3, 5, 8, 13}

_RULES: dict[tuple, int] = {
    ("bug",     "high"):   5,
    ("bug",     "medium"): 3,
    ("bug",     "low"):    2,
    ("feature", "high"):   8,
    ("feature", "medium"): 5,
    ("feature", "low"):    3,
    ("research","high"):   8,
    ("research","medium"): 5,
    ("research","low"):    3,
    ("admin",   "high"):   3,
    ("admin",   "medium"): 2,
    ("admin",   "low"):    1,
    ("other",   "high"):   5,
    ("other",   "medium"): 3,
    ("other",   "low"):    2,
}

# Complexity boosts based on keywords in task text
_COMPLEX_KEYWORDS = ["refactor", "migrate", "redesign", "rewrite", "architecture", "integration"]
_SIMPLE_KEYWORDS  = ["update", "fix typo", "rename", "add comment", "bump version"]


def estimate_story_points(task: dict) -> int:
    """
    Estimate story points using rule table + keyword adjustment.
    Always returns a Fibonacci number.
    """
    task_type = task.get("type", "other").lower()
    priority  = task.get("priority", "medium").lower()
    task_text = task.get("task", "").lower()

    base = _RULES.get((task_type, priority), 3)

    # Adjust for complexity keywords
    if any(kw in task_text for kw in _COMPLEX_KEYWORDS):
        base = min(base * 2, 13)
    elif any(kw in task_text for kw in _SIMPLE_KEYWORDS):
        base = max(base // 2, 1)

    # Snap to nearest Fibonacci
    pts = min(FIBONACCI, key=lambda f: abs(f - base))
    print(f"[estimator] Rule-based: {pts} pts ({task_type}/{priority})")
    return pts
