#!/usr/bin/env bash
#
# Give the service user read-only access to one private GitHub repository.
#
#   sudo bash deploy/add-deploy-key.sh https://github.com/Sosnowsky/fusion_scripts.git
#
# A deploy key must be unique across the whole of GitHub, so this generates a
# fresh key per repository and an ssh Host alias to keep them apart. Re-running
# it for a repository that already has a key just re-verifies access.
#
# Past three or four repositories, stop: a machine user, or a fine-grained token
# with read access to all of them, is one credential instead of N.

set -euo pipefail

SERVICE_USER=${SERVICE_USER:-fusionui}
URL=${1:-}

[[ $EUID -eq 0 ]] || { echo "Run me as root: sudo bash $0 <repo-url>" >&2; exit 1; }
[[ -n "$URL" ]] || { echo "Usage: sudo bash $0 https://github.com/<owner>/<repo>.git" >&2; exit 1; }

# https://github.com/owner/repo.git or git@github.com:owner/repo.git -> owner/repo
SLUG=$(sed -E 's#^(https://github\.com/|git@github\.com:)##; s#\.git$##' <<<"$URL")
[[ "$SLUG" == */* ]] || { echo "Could not read owner/repo out of $URL" >&2; exit 1; }
REPO=${SLUG#*/}
ALIAS="github-$REPO"
HOME_DIR=$(getent passwd "$SERVICE_USER" | cut -d: -f6)
KEY="$HOME_DIR/.ssh/id_$(tr -c 'a-zA-Z0-9' '_' <<<"$REPO" | sed 's/_*$//')"

run() { sudo -u "$SERVICE_USER" -H "$@"; }

if [[ -f "$KEY" ]]; then
  echo "Key already exists: $KEY"
else
  run install -d -m 700 "$HOME_DIR/.ssh"
  run ssh-keygen -q -t ed25519 -N "" -C "fusion-ui@$(hostname -f):$SLUG" -f "$KEY"
  echo "Generated $KEY"
fi

if ! run grep -q "^Host $ALIAS\$" "$HOME_DIR/.ssh/config" 2>/dev/null; then
  run tee -a "$HOME_DIR/.ssh/config" >/dev/null <<CONFIG

Host $ALIAS
    HostName github.com
    User git
    IdentityFile $KEY
    IdentitiesOnly yes
CONFIG
  run chmod 600 "$HOME_DIR/.ssh/config"
fi

# IdentitiesOnly above is what makes this safe: without it ssh offers every key
# it holds and GitHub answers for whichever one it recognises first, which can
# be a different repository entirely.
run git config --global "url.$ALIAS:$SLUG.git.insteadOf" "https://github.com/$SLUG.git"

if ! run grep -q '^github\.com ' "$HOME_DIR/.ssh/known_hosts" 2>/dev/null; then
  run sh -c "ssh-keyscan github.com >> $HOME_DIR/.ssh/known_hosts" 2>/dev/null
  echo
  echo "Recorded github.com's host keys. Compare these against the fingerprints"
  echo "GitHub publishes before relying on them:"
  run ssh-keygen -lf "$HOME_DIR/.ssh/known_hosts" | sed 's/^/    /'
fi

cat <<MSG

Add this as a read-only deploy key:

    https://github.com/$SLUG/settings/keys/new

  Title:  $(hostname -f)
  Key:    (below)
  Allow write access: leave UNCHECKED

MSG
run cat "$KEY.pub"
echo
read -rp "Press enter once the key is added … "

if GIT_TERMINAL_PROMPT=0 run git ls-remote "https://github.com/$SLUG.git" >/dev/null 2>&1; then
  echo "OK — $SERVICE_USER can now read $SLUG."
else
  echo "Still no access to $SLUG." >&2
  echo "Check the key was added to that repository, and that it is the whole" >&2
  echo "line from $KEY.pub." >&2
  exit 1
fi
