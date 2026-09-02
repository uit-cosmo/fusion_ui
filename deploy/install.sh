#!/usr/bin/env bash
#
# Install or update Shot Explorer on the group server. Run as root:
#
#   sudo bash deploy/install.sh
#
# Every step is idempotent, so re-running it after a `git pull` is the normal
# way to deploy an update. Override any of the settings below on the command
# line, e.g.  sudo STATE_DIR=/hdd2/fusion_ui bash deploy/install.sh
#
# What it will NOT do without you: set the shared password (step 6 prompts),
# fill in .env (step 3 opens it in an editor), or touch the firewall unless
# CAMPUS_SUBNET is set.

set -euo pipefail

# Never let git stop at an interactive credential prompt: this script may run
# unattended, and GitHub has not accepted account passwords for git operations
# since 2021, so there is nothing useful to type anyway. A private repository
# fails immediately instead, and the handler below says how to grant access.
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

APP_DIR=${APP_DIR:-/opt/fusion-ui}
SRC_DIR=${SRC_DIR:-/opt/src}                 # sibling checkouts live here
STATE_DIR=${STATE_DIR:-/hdd1/fusion_ui}      # SQLite file + result cache
SERVICE_USER=${SERVICE_USER:-fusionui}
REPO_URL=${REPO_URL:-https://github.com/Sosnowsky/fusion_ui.git}
BRANCH=${BRANCH:-main}
SERVER_NAME=${SERVER_NAME:-$(hostname -f)}
CAMPUS_SUBNET=${CAMPUS_SUBNET:-}             # e.g. 10.228.0.0/16; empty = ask, offering
                                             # the subnet the server is itself on
HTPASSWD_USER=${HTPASSWD_USER:-fusion}

# name=url for the packages that are not on PyPI. A checkout already present in
# $SRC_DIR is used as it stands and never overwritten, so a private repository
# the server cannot reach is handled by copying it in first:
#
#   rsync -a --exclude .venv ~/Git/experimental_database server:/tmp/
#   sudo mv /tmp/experimental_database $SRC_DIR/
#
# The durable fix is a read-only deploy key on the server for that repository,
# which lets this script update it like the others.
DEPENDENCIES=(
  "imaging-methods=https://github.com/uit-cosmo/phantom.git"
  "experimental_database=https://github.com/uit-cosmo/experimental_database.git"
  "fusion_scripts=https://github.com/Sosnowsky/fusion_scripts.git"
  "velocity-estimation=https://github.com/uit-cosmo/velocity-estimation.git"
  "fpp-analysis-tools=https://github.com/uit-cosmo/fpp-analysis-tools.git"
)

PIP="$APP_DIR/.venv/bin/pip"
FUSION_UI="$APP_DIR/.venv/bin/fusion-ui"
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# The network the server itself sits on -- the laptops are almost always on it
# too. Offered as a default, never applied silently: a firewall rule is about
# who gets in, and the interface prefix can be wider than you meant.
detect_subnet() {
  local iface cidr
  iface=$(ip -4 route show default | awk '{print $5; exit}') || return 1
  cidr=$(ip -4 -o addr show dev "$iface" | awk '{print $4; exit}') || return 1
  [[ -n "$cidr" ]] || return 1
  python3 -c 'import ipaddress,sys; print(ipaddress.ip_network(sys.argv[1], strict=False))' "$cidr"
}
as_service_user() { sudo -u "$SERVICE_USER" "$@"; }

[[ $EUID -eq 0 ]] || { echo "Run me as root: sudo bash $0" >&2; exit 1; }

step "1. Packages and service account"
apt-get update -qq
apt-get install -y --no-install-recommends \
  git python3-venv nginx apache2-utils openssl curl
id -u "$SERVICE_USER" &>/dev/null ||
  useradd --system --create-home --home-dir "/var/lib/$SERVICE_USER" "$SERVICE_USER"

step "2. Checkouts"
mkdir -p "$SRC_DIR"
chown "$SERVICE_USER:$SERVICE_USER" "$SRC_DIR"
for entry in "${DEPENDENCIES[@]}"; do
  name=${entry%%=*}; url=${entry#*=}; target="$SRC_DIR/$name"
  if [[ -d "$target/.git" ]]; then
    if as_service_user git -C "$target" pull --ff-only --quiet 2>/dev/null; then
      echo "  $name: updated"
    else
      # No credentials for a private repository, a detached HEAD, local edits.
      # Whatever is on disk is what the maintainer put there; use it.
      echo "  $name: could not update, using the checkout as it stands"
    fi
  elif [[ -d "$target" ]]; then
    echo "  $name: copied in by hand, using as is"
  else
    echo "  $name: cloning"
    if ! as_service_user git clone --depth 50 --quiet "$url" "$target"; then
      cat >&2 <<MSG

  Could not clone $name from $url.

  If it is private, this server has no access to it. Either give it one --
  a read-only deploy key for the service user, see deploy/README.md -- or
  copy the checkout in and re-run; an existing directory is used as it stands:

      rsync -a --exclude .venv <your machine>:path/to/$name /tmp/
      sudo mv /tmp/$name $SRC_DIR/

MSG
      exit 1
    fi
  fi
  # pip install -e writes .egg-info into the source tree, so the service user
  # needs to own it -- including anything copied in as root.
  chown -R "$SERVICE_USER:$SERVICE_USER" "$target"
done

mkdir -p "$APP_DIR"
chown "$SERVICE_USER:$SERVICE_USER" "$APP_DIR"
if [[ -d "$APP_DIR/.git" ]]; then
  as_service_user git -C "$APP_DIR" fetch origin "$BRANCH"
  as_service_user git -C "$APP_DIR" checkout "$BRANCH"
  as_service_user git -C "$APP_DIR" pull --ff-only
elif ! as_service_user git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"; then
  cat >&2 <<MSG

  Could not clone the app from $REPO_URL (branch $BRANCH).

  If the repository is private, the service user needs its own read-only
  deploy key -- one key grants access to exactly one repository, so this
  needs a separate one from any dependency key. See deploy/README.md.

  Then re-run with the SSH URL:

      sudo REPO_URL=git@github.com:Sosnowsky/fusion_ui.git bash $0

MSG
  exit 1
fi

step "3. Configuration"
mkdir -p "$STATE_DIR/cache"
chown -R "$SERVICE_USER:$SERVICE_USER" "$STATE_DIR"
if [[ ! -f "$APP_DIR/.env" ]]; then
  as_service_user cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  as_service_user sed -i \
    -e "s|^FUSION_UI_DB=.*|FUSION_UI_DB=$STATE_DIR/shot_explorer.sqlite|" \
    -e "s|^FUSION_UI_CACHE=.*|FUSION_UI_CACHE=$STATE_DIR/cache|" \
    "$APP_DIR/.env"
  echo
  echo "  Set FUSION_DISCHARGE_DB and FUSION_DATA_FOLDER for this server."
  read -rp "  Press enter to open $APP_DIR/.env … "
  as_service_user "${EDITOR:-nano}" "$APP_DIR/.env"
else
  echo "  $APP_DIR/.env exists, leaving it alone."
fi

step "4. Python environment"
[[ -x "$PIP" ]] || as_service_user python3 -m venv "$APP_DIR/.venv"
as_service_user "$PIP" install --quiet --upgrade pip
# The local checkouts must go in first: they are pinned in pyproject.toml but
# are not on PyPI, so installing the app alone cannot resolve them.
for entry in "${DEPENDENCIES[@]}"; do
  as_service_user "$PIP" install --quiet -e "$SRC_DIR/${entry%%=*}"
done
as_service_user "$PIP" install --quiet -e "$APP_DIR"

step "5. Database and first index"
as_service_user "$FUSION_UI" init-db
as_service_user "$FUSION_UI" rescan
as_service_user "$FUSION_UI" status

echo "  Keeping the index fresh (new shots appear only after a rescan):"
cron_line="*/15 * * * * $FUSION_UI rescan >> /var/log/fusion-ui-rescan.log 2>&1"
if as_service_user crontab -l 2>/dev/null | grep -qF "$FUSION_UI rescan"; then
  echo "  cron entry already present"
else
  { as_service_user crontab -l 2>/dev/null || true; echo "$cron_line"; } |
    as_service_user crontab -
  echo "  added: $cron_line"
fi
touch /var/log/fusion-ui-rescan.log
chown "$SERVICE_USER:$SERVICE_USER" /var/log/fusion-ui-rescan.log

step "6. systemd"
sed -e "s|/opt/fusion-ui|$APP_DIR|g" \
    -e "s|^User=.*|User=$SERVICE_USER|" \
    -e "s|^Group=.*|Group=$SERVICE_USER|" \
    -e "s|^ReadWritePaths=.*|ReadWritePaths=$STATE_DIR|" \
    "$APP_DIR/deploy/fusion-ui.service" > /etc/systemd/system/fusion-ui.service
systemctl daemon-reload
systemctl enable --now fusion-ui
systemctl restart fusion-ui
sleep 3
systemctl is-active --quiet fusion-ui ||
  { journalctl -u fusion-ui -n 30 --no-pager; exit 1; }
echo "  local: HTTP $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8501/)"

step "7. nginx, TLS and the shared password"
if [[ ! -f /etc/nginx/.htpasswd ]]; then
  echo "  Creating the shared password for user '$HTPASSWD_USER':"
  htpasswd -c /etc/nginx/.htpasswd "$HTPASSWD_USER"
else
  echo "  /etc/nginx/.htpasswd exists; run 'htpasswd /etc/nginx/.htpasswd $HTPASSWD_USER' to change it."
fi
if [[ ! -f /etc/nginx/ssl/fusion-ui.crt ]]; then
  mkdir -p /etc/nginx/ssl
  openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
    -keyout /etc/nginx/ssl/fusion-ui.key -out /etc/nginx/ssl/fusion-ui.crt \
    -subj "/CN=$SERVER_NAME" 2>/dev/null
fi
sed "s|server_name _;|server_name $SERVER_NAME;|" \
  "$APP_DIR/deploy/nginx.conf" > /etc/nginx/sites-available/fusion-ui
ln -sf /etc/nginx/sites-available/fusion-ui /etc/nginx/sites-enabled/fusion-ui
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

step "8. Firewall"
if ufw status 2>/dev/null | grep -q '443.*ALLOW'; then
  echo "  A rule for 443 already exists:"
  ufw status | grep '443' | sed 's/^/    /'
elif [[ -z "$CAMPUS_SUBNET" ]]; then
  detected=$(detect_subnet 2>/dev/null || true)
  echo "  This server is on ${detected:-an undetermined network}."
  read -rp "  Allow 443 from [${detected:-none}] (or type a subnet, or 'skip'): " reply
  CAMPUS_SUBNET=${reply:-$detected}
  [[ "$CAMPUS_SUBNET" == "skip" || "$CAMPUS_SUBNET" == "none" ]] && CAMPUS_SUBNET=""
fi

if [[ -n "$CAMPUS_SUBNET" ]]; then
  ufw allow from "$CAMPUS_SUBNET" to any port 443 proto tcp
  ufw status | sed 's/^/    /'
else
  echo "  Skipped. To open it later:"
  echo "    sudo ufw allow from <subnet> to any port 443 proto tcp"
fi

step "9. Verify"
bash "$APP_DIR/deploy/verify.sh" "$SERVER_NAME"
