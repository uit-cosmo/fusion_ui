#!/usr/bin/env bash
#
# Install or update Shot Explorer on the group server. Run as root:
#
#   sudo bash deploy/install.sh --branch phase-00-skeleton
#
# Every step is idempotent, so re-running it after a `git pull` is the normal
# way to deploy an update. `--help` lists every setting; each also works as an
# environment variable, though whether sudo passes those through depends on the
# sudoers config, so the flags are the reliable form.
#
# What it will NOT do without you: fill in .env (step 3 opens an editor), set
# the shared password (step 7 prompts), or change the firewall (step 8 asks).

set -euo pipefail

# Never let git stop at an interactive credential prompt: this script may run
# unattended, and GitHub has not accepted account passwords for git operations
# since 2021, so there is nothing useful to type anyway. A private repository
# fails immediately instead, and the handler below says how to grant access.
export GIT_TERMINAL_PROMPT=0
export GIT_SSH_COMMAND="ssh -o BatchMode=yes"

# git 2.43 against a curl built on GnuTLS gets a spurious "401 Basic realm"
# from GitHub on the protocol-v2 POST to /git-upload-pack when it re-uses the
# multiplexed HTTP/2 connection from the ref advertisement -- which turns a
# public, anonymous clone into a credential prompt. Pinning the transport to
# HTTP/1.1 avoids it and keeps protocol v2. Passed as GIT_CONFIG_* rather than
# a config file so it applies to every git call here and changes nothing
# permanent on the host.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=http.version
export GIT_CONFIG_VALUE_0=HTTP/1.1

# sudo does not forward the environment, so the variables above have to be
# handed to git explicitly or they are silently lost.
GIT_ENV=(
  GIT_TERMINAL_PROMPT="$GIT_TERMINAL_PROMPT"
  GIT_SSH_COMMAND="$GIT_SSH_COMMAND"
  GIT_CONFIG_COUNT="$GIT_CONFIG_COUNT"
  GIT_CONFIG_KEY_0="$GIT_CONFIG_KEY_0"
  GIT_CONFIG_VALUE_0="$GIT_CONFIG_VALUE_0"
)

APP_DIR=${APP_DIR:-/opt/fusion-ui}
SRC_DIR=${SRC_DIR:-/opt/src}                 # sibling checkouts live here
STATE_DIR=${STATE_DIR:-/hdd1/fusion_ui}      # SQLite file + result cache
SERVICE_USER=${SERVICE_USER:-fusionui}
REPO_URL=${REPO_URL:-https://github.com/uit-cosmo/fusion_ui.git}
BRANCH=${BRANCH:-main}
SERVER_NAME=${SERVER_NAME:-$(hostname -f)}
CAMPUS_SUBNET=${CAMPUS_SUBNET:-}             # e.g. 10.228.0.0/16; empty = ask, offering
                                             # the subnet the server is itself on
HTPASSWD_USER=${HTPASSWD_USER:-fusion}

# name=url for the packages that are not on PyPI. Only experimental_database and
# fusion_scripts are private; the rest clone anonymously and need no credentials.
# A checkout already present in
# $SRC_DIR is used as it stands and never overwritten, so a private repository
# the server cannot reach is handled by copying it in first:
#
#   rsync -a --exclude .venv ~/Git/experimental_database server:/tmp/
#   sudo mv /tmp/experimental_database $SRC_DIR/
#
# The durable fix is a read-only deploy key on the server for that repository,
# which lets this script update it like the others.
DEPENDENCIES=(
  "imaging-methods=https://github.com/uit-cosmo/imaging-methods.git"
  "experimental_database=https://github.com/uit-cosmo/experimental_database.git"
  "fusion_scripts=https://github.com/Sosnowsky/fusion_scripts.git"
  "velocity-estimation=https://github.com/uit-cosmo/velocity-estimation.git"
  "fpp-analysis-tools=https://github.com/uit-cosmo/fpp-analysis-tools.git"
)

usage() {
  cat <<USAGE
Install or update Shot Explorer. Run as root.

Usage: sudo bash $0 [options]

  --branch NAME       branch to deploy                 [$BRANCH]
  --repo-url URL      where to clone the app from      [$REPO_URL]
  --app-dir PATH      the checkout to run from         [$APP_DIR]
  --src-dir PATH      where dependency checkouts live  [$SRC_DIR]
  --state-dir PATH    SQLite file and result cache     [$STATE_DIR]
  --user NAME         service account                  [$SERVICE_USER]
  --server-name NAME  hostname in the certificate/nginx[$SERVER_NAME]
  --subnet CIDR       allow 443 from this range        [${CAMPUS_SUBNET:-ask}]
  --htpasswd-user U   shared login name                [$HTPASSWD_USER]
  -h, --help          this text

Each option also reads from the matching environment variable
(BRANCH, REPO_URL, APP_DIR, SRC_DIR, STATE_DIR, SERVICE_USER,
SERVER_NAME, CAMPUS_SUBNET, HTPASSWD_USER).
USAGE
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --branch)        BRANCH=$2; shift 2 ;;
    --repo-url)      REPO_URL=$2; shift 2 ;;
    --app-dir)       APP_DIR=$2; shift 2 ;;
    --src-dir)       SRC_DIR=$2; shift 2 ;;
    --state-dir)     STATE_DIR=$2; shift 2 ;;
    --user)          SERVICE_USER=$2; shift 2 ;;
    --server-name)   SERVER_NAME=$2; shift 2 ;;
    --subnet)        CAMPUS_SUBNET=$2; shift 2 ;;
    --htpasswd-user) HTPASSWD_USER=$2; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *)               echo "Unknown option: $1" >&2; echo >&2; usage >&2; exit 1 ;;
  esac
done

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
as_service_user() { sudo -u "$SERVICE_USER" env "${GIT_ENV[@]}" "$@"; }

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

# Fail with the list of branches rather than git's bare "couldn't find remote
# ref", which does not hint that --branch is the knob.
if ! as_service_user git ls-remote --exit-code --heads "$REPO_URL" "$BRANCH" >/dev/null 2>&1; then
  echo >&2
  echo "  No branch '$BRANCH' on $REPO_URL. Available:" >&2
  as_service_user git ls-remote --heads "$REPO_URL" 2>/dev/null \
    | sed 's#.*refs/heads/#    #' >&2 \
    || echo "    (could not list -- is the repository reachable?)" >&2
  echo >&2
  echo "  Pick one with:  sudo bash $0 --branch <name>" >&2
  exit 1
fi

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

      sudo bash $0 --repo-url git@github.com:uit-cosmo/fusion_ui.git

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
# nginx binds every listen directive at startup, so one taken port stops the
# whole service -- including the 443 vhost that is the actual deployment.
if ss -tln '( sport = :80 )' 2>/dev/null | grep -q LISTEN &&
   ! systemctl is-active --quiet nginx; then
  holder=$(ss -tlnp '( sport = :80 )' 2>/dev/null |
           sed -n 's/.*users:(("\([^"]*\)".*/\1/p' | head -1)
  echo "  Port 80 is held by ${holder:-another process}; serving 443 only."
  echo "  http://$SERVER_NAME will keep going wherever it goes today."
  redirect_fix='/# BEGIN http-redirect/,/# END http-redirect/d'
else
  redirect_fix=''
fi

nginx_version=$(nginx -v 2>&1 | sed 's|.*/||')
if [[ "$(printf '%s\n' 1.25.1 "$nginx_version" | sort -V | head -1)" == "1.25.1" ]]; then
  # 1.25.1+ deprecated the listen parameter in favour of its own directive.
  http2_fix='s|listen 443 ssl http2;|listen 443 ssl;\n    http2 on;|'
else
  http2_fix=''
  echo "  nginx $nginx_version: keeping the legacy 'listen ... http2' form."
fi
sed -e "s|server_name _;|server_name $SERVER_NAME;|" -e "$http2_fix" \
  -e "$redirect_fix" \
  "$APP_DIR/deploy/nginx.conf" > /etc/nginx/sites-available/fusion-ui
ln -sf /etc/nginx/sites-available/fusion-ui /etc/nginx/sites-enabled/fusion-ui
rm -f /etc/nginx/sites-enabled/default
if ! nginx -t; then
  echo >&2
  echo "  The nginx site is invalid -- see the error above. nginx has been left" >&2
  echo "  as it was; fix /etc/nginx/sites-available/fusion-ui and re-run." >&2
  exit 1
fi
# enable --now rather than reload: a previous failed `nginx -t` leaves the
# service stopped, and reload cannot start a dead service.
systemctl enable --now nginx
systemctl reload nginx

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
  if ufw status 2>/dev/null | grep -qi 'inactive'; then
    echo "  Note: ufw is inactive, so this rule filters nothing yet. To turn it"
    echo "  on without locking yourself out of ssh:"
    echo "    sudo ufw allow OpenSSH && sudo ufw enable"
  fi
else
  echo "  Skipped. To open it later:"
  echo "    sudo ufw allow from <subnet> to any port 443 proto tcp"
fi

step "9. Verify"
verify_status=0
bash "$APP_DIR/deploy/verify.sh" "$SERVER_NAME" || verify_status=$?

step "Done"
printf '  Open \033[1mhttps://%s/\033[0m and log in as "%s".\n' \
  "$SERVER_NAME" "$HTPASSWD_USER"
if [[ $verify_status -ne 0 ]]; then
  echo "  (That is the address once the failing checks above are fixed.)"
fi
exit $verify_status
