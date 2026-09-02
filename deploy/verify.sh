#!/usr/bin/env bash
#
# Check a running deployment. Safe to run any time, from the server or a laptop:
#
#   bash deploy/verify.sh <host> [user]
#
# The websocket handshake is the check that matters. Streamlit talks over a
# websocket, so a proxy that drops the Upgrade header serves a page that loads
# and then hangs forever with no error message -- a plain 200 proves nothing.

set -uo pipefail
HOST=${1:-$(hostname -f)}
USER_NAME=${2:-fusion}
FAILED=0

http_code() {  # curl already prints 000 when it cannot connect
  curl -sk -o /dev/null -w '%{http_code}' --max-time 10 "$@" 2>/dev/null || true
}

check() {  # check <description> <expected> <actual>
  if [[ "$3" == "$2" ]]; then
    printf '  \033[32mok\033[0m    %-28s %s\n' "$1" "$3"
  else
    printf '  \033[31mFAIL\033[0m  %-28s %s (expected %s)\n' "$1" "$3" "$2"
    FAILED=1
  fi
}

read -rsp "Shared password for '$USER_NAME': " PASSWORD; echo; echo

check "streamlit on loopback" "200" "$(http_code http://127.0.0.1:8501/)"
check "https page"           "200" "$(http_code -u "$USER_NAME:$PASSWORD" "https://$HOST/")"
check "password required"    "401" "$(http_code "https://$HOST/")"

# 101 Switching Protocols, or the page loads and then hangs with no error.
handshake=$(curl -sk -i -N --max-time 10 \
  -u "$USER_NAME:$PASSWORD" \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: $(openssl rand -base64 16)" \
  "https://$HOST/_stcore/stream" 2>/dev/null | head -1 | grep -oE '[0-9]{3}' || echo 000)
check "websocket handshake"  "101" "$handshake"

echo
if [[ $FAILED -eq 0 ]]; then
  echo "All good — https://$HOST/ is ready for the group."
else
  echo "Something is wrong. Start with:  sudo journalctl -u fusion-ui -n 50"
  if [[ "$handshake" == "200" ]]; then
    echo
    echo "The handshake returned 200, not 101: nginx is swallowing the Upgrade"
    echo "header. The page will load and then hang for everyone. Check the"
    echo "proxy_set_header Upgrade/Connection lines in the site config."
  fi
fi
exit $FAILED
