# lit-scout setup steps for Claude Code

These steps are for Claude Code to follow when a person asks it to set up lit-scout. Follow them in order. Never print, echo or repeat an API key.

1. Install the plugin:

   ```
   claude plugin marketplace add flaviokicis/lit-scout
   claude plugin install lit-scout@lit-scout
   ```

   If either command fails, show the person the error and stop.

2. Check the tools. `python3` must exist. If `pdftotext` is missing, run `brew install poppler` on macOS, or tell the person to run `sudo apt install poppler-utils` on Linux.

3. Open the OpenAlex key page: `open https://openalex.org/settings/api` on macOS, `xdg-open https://openalex.org/settings/api` on Linux.

4. Show the person this message exactly, with one empty line above it and one below it. Then stop and wait for their reply.

   ```
   ==============================================================

      Sign up or log in on the OpenAlex page that just opened.
      Click the link in the email OpenAlex sends you.
      Then copy your API key.

      DO NOT PASTE YOUR KEY HERE.
   ```
   **Just type: READY**
   ```
   ==============================================================
   ```

5. When the person types READY, save the key from the clipboard. On Linux, use `xclip -o -selection clipboard` or `wl-paste` in place of `pbpaste`.

   ```
   pbpaste | python3 -c 'import json,sys; k=sys.stdin.read().strip(); assert k and " " not in k and len(k)<200, "clipboard does not hold a key"; print(json.dumps({"openalexKey": k}))' | claude plugin configure lit-scout@lit-scout --values-stdin
   ```

   If the check fails, tell the person to copy the key again and type READY again.

6. Test the key. This prints only the HTTP code, which must be `200`:

   ```
   curl -s -o /dev/null -w "%{http_code}\n" "https://api.openalex.org/works?search=test&per-page=1&api_key=$(pbpaste | tr -d '[:space:]')"
   ```

   A `401` or `403` usually means the email link was not clicked yet.

7. Clear the clipboard: `pbcopy < /dev/null`.

8. Tell the person that lit-scout is ready, and that they must restart Claude Code with `claude --resume` to load it. Then mention that a Semantic Scholar key is optional and goes in `/config` under lit-scout.
