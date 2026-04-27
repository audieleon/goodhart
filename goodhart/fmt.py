"""Terminal formatting utilities.

Provides colored, structured output for CLI. Detects whether the
terminal supports color (isatty) and degrades gracefully to plain
text for pipes and CI.
"""

import os
import sys
import textwrap


def _supports_color():
    """Check if stdout supports ANSI color codes."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_COLOR = _supports_color()

# ANSI escape codes
_RESET = "\033[0m" if _COLOR else ""
_BOLD = "\033[1m" if _COLOR else ""
_DIM = "\033[2m" if _COLOR else ""
_RED = "\033[31m" if _COLOR else ""
_YELLOW = "\033[33m" if _COLOR else ""
_GREEN = "\033[32m" if _COLOR else ""
_BLUE = "\033[34m" if _COLOR else ""
_CYAN = "\033[36m" if _COLOR else ""
_MAGENTA = "\033[35m" if _COLOR else ""
_WHITE = "\033[37m" if _COLOR else ""

# Semantic colors
CRITICAL_COLOR = f"{_BOLD}{_RED}"
WARNING_COLOR = f"{_BOLD}{_YELLOW}"
INFO_COLOR = f"{_BLUE}"
RULE_COLOR = f"{_CYAN}"
REC_COLOR = f"{_GREEN}"
DIM_COLOR = _DIM
HEADER_COLOR = _BOLD
RESET = _RESET

# Fixed terminal width for wrapping
try:
    TERM_WIDTH = min(os.get_terminal_size().columns, 100) if sys.stdout.isatty() else 80
except (OSError, ValueError, AttributeError):
    TERM_WIDTH = 80
INDENT = "     "
WRAP_WIDTH = max(TERM_WIDTH - len(INDENT), 20)


def header(title, subtitle=None):
    """Print a bold header bar."""
    line = "=" * TERM_WIDTH
    print(f"\n{HEADER_COLOR}{line}{RESET}")
    print(f"{HEADER_COLOR}  {title}{RESET}")
    if subtitle:
        print(f"  {DIM_COLOR}{subtitle}{RESET}")
    print(f"{HEADER_COLOR}{line}{RESET}\n")


def section(title, count=None):
    """Print a section header like 'CRITICAL (3):'."""
    count_str = f" ({count})" if count is not None else ""
    print(f"{HEADER_COLOR}{title}{count_str}:{RESET}\n")


def verdict(v, verbose=False):
    """Print a single verdict with color and structure."""
    sev = v.severity.value
    if sev == "critical":
        icon = f"{CRITICAL_COLOR}X{RESET}"
        sev_color = CRITICAL_COLOR
    elif sev == "warning":
        icon = f"{WARNING_COLOR}!{RESET}"
        sev_color = WARNING_COLOR
    else:
        icon = f"{INFO_COLOR}i{RESET}"
        sev_color = INFO_COLOR

    rule = f"{RULE_COLOR}{v.rule_name}{RESET}"

    # Main message — wrap long lines
    msg_lines = textwrap.wrap(v.message, width=WRAP_WIDTH)
    print(f"  [{icon}] [{rule}]")
    for line in msg_lines:
        print(f"{INDENT}{line}")

    # Recommendation
    if v.recommendation:
        rec_lines = textwrap.wrap(v.recommendation, width=WRAP_WIDTH - 3)
        print(f"\n{INDENT}{REC_COLOR}-> {rec_lines[0]}{RESET}")
        for line in rec_lines[1:]:
            print(f"{INDENT}   {REC_COLOR}{line}{RESET}")

    # Learn more (verbose only)
    if verbose and v.learn_more:
        print()
        for paragraph in v.learn_more.split("\n"):
            wrapped = textwrap.wrap(paragraph.strip(), width=WRAP_WIDTH - 3)
            for line in wrapped:
                print(f"{INDENT}   {DIM_COLOR}{line}{RESET}")
            if wrapped:
                print()

    print()  # blank line between verdicts


GREEN_LABEL = f"{_GREEN}" if _COLOR else ""


def summary(n_critical, n_warning, n_info):
    """Print the summary line."""
    parts = []
    if n_critical:
        parts.append(f"{CRITICAL_COLOR}{n_critical} critical{RESET}")
    else:
        parts.append(f"{GREEN_LABEL}0 critical{RESET}")
    parts.append(f"{n_warning} warnings")
    parts.append(f"{n_info} info")
    print(f"{DIM_COLOR}{'─' * TERM_WIDTH}{RESET}")
    print(f"  {', '.join(parts)}")
    print()


def passed_banner():
    """Print a green PASSED banner."""
    print(f"  {_GREEN}{_BOLD}PASSED{RESET} — no structural reward traps detected.\n")


def failed_banner(n):
    """Print a red FAILED banner."""
    print(f"  {CRITICAL_COLOR}FAILED{RESET} — {n} critical issue(s) found.\n")


def rule_list_item(name, description, width=32):
    """Print a rule in the --rules listing."""
    # Wrap description to fit after the name column
    desc_width = TERM_WIDTH - width - 6
    desc_lines = textwrap.wrap(description, width=desc_width)
    print(f"    {RULE_COLOR}{name:<{width}}{RESET} {desc_lines[0] if desc_lines else ''}")
    for line in desc_lines[1:]:
        print(f"    {' ' * width} {line}")


def explain_header(rule_name, description):
    """Print the --explain header."""
    line = "=" * TERM_WIDTH
    print(f"\n{HEADER_COLOR}{line}{RESET}")
    print(f"  {HEADER_COLOR}Rule:{RESET} {RULE_COLOR}{rule_name}{RESET}")
    print(f"  {description}")
    print(f"{HEADER_COLOR}{line}{RESET}\n")


def explain_section(title, content):
    """Print a section in --explain output."""
    print(f"  {HEADER_COLOR}{title}:{RESET}")
    if isinstance(content, str):
        for paragraph in content.split("\n"):
            wrapped = textwrap.wrap(paragraph.strip(), width=WRAP_WIDTH - 3)
            for line in wrapped:
                print(f"     {line}")
            if wrapped:
                print()
    elif isinstance(content, list):
        for item in content:
            print(f"     - {item}")
        print()


def category_header(name):
    """Print a category header in --rules."""
    print(f"\n  {HEADER_COLOR}{name}{RESET}")
    print(f"  {DIM_COLOR}{'─' * (TERM_WIDTH - 2)}{RESET}")
