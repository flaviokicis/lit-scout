#!/usr/bin/env bash
# Reports what lit-scout needs on this machine. Prints no key values. Always exits 0.

row() { printf '%-24s %-9s %s\n' "$1" "$2" "$3"; }

install_hint_poppler() {
  if [ "$(uname)" = "Darwin" ]; then echo "brew install poppler"; else echo "apt install poppler-utils"; fi
}

# Names of options the plugin reports as unset; empty when the plugin is not installed or the output has no such list.
unset_options() {
  claude plugin configure lit-scout@lit-scout --json 2>/dev/null | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
def walk(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if "unset" in key.lower() and isinstance(value, list):
                for item in value:
                    print(item if isinstance(item, str) else item.get("key", item.get("name", "")))
            else:
                walk(value)
    elif isinstance(node, list):
        for item in node:
            walk(item)
walk(data)
' 2>/dev/null
}

UNSET="$(unset_options)"

is_set() { # option name, env variable name
  if [ -n "${!2:-}" ]; then return 0; fi
  if [ -n "$UNSET" ] && ! printf '%s\n' "$UNSET" | grep -qx "$1"; then return 0; fi
  return 1
}

if command -v python3 >/dev/null 2>&1; then row python3 OK "$(python3 --version 2>&1)"; else row python3 MISSING "install Python 3 from https://www.python.org/downloads/"; fi

if command -v pdftotext >/dev/null 2>&1; then row pdftotext OK "full-text reading"; else row pdftotext MISSING "$(install_hint_poppler)"; fi

if command -v gh >/dev/null 2>&1; then row gh OK "GitHub source"; else row gh OPTIONAL "GitHub source: brew install gh  (or: apt install gh)"; fi

if is_set openalexKey OPENALEX_API_KEY; then
  if [ -n "${OPENALEX_API_KEY:-}" ] && command -v curl >/dev/null 2>&1; then
    code="$(printf 'url = "https://api.openalex.org/works?per-page=1&api_key=%s"\n' "$OPENALEX_API_KEY" | curl -s -o /dev/null -m 20 -w '%{http_code}' -K - 2>/dev/null)"
    row "OpenAlex key" OK "set; test request returned HTTP ${code:-000}"
  else
    row "OpenAlex key" OK "set in plugin options (not tested)"
  fi
else
  row "OpenAlex key" OPTIONAL "free key at https://openalex.org/settings/api; then: export OPENALEX_API_KEY=... or claude plugin configure lit-scout@lit-scout"
fi

if is_set s2Key S2_API_KEY; then
  row "Semantic Scholar key" OK "set"
else
  row "Semantic Scholar key" OPTIONAL "free key at https://www.semanticscholar.org/product/api; then: export S2_API_KEY=... or claude plugin configure lit-scout@lit-scout"
fi

exit 0
