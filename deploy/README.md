# Deploying Shot Explorer

**The short version:**

```bash
git clone https://github.com/Sosnowsky/fusion_ui.git /tmp/fusion-ui
sudo bash /tmp/fusion-ui/deploy/install.sh
```

`install.sh` is every step below, in order and idempotent — re-run it after a
`git pull` and it updates in place. It stops three times for you: to edit `.env`,
to set the shared password, and to confirm the firewall rule, for which it
offers the subnet the server is itself on. Everything else has a default you can
override on the command line (`APP_DIR`, `SRC_DIR`, `STATE_DIR`, `SERVICE_USER`,
`REPO_URL`, `BRANCH`, `SERVER_NAME`, `CAMPUS_SUBNET`).

To check a deployment at any time, from the server or a laptop:

```bash
bash deploy/verify.sh <host>
```

The rest of this file is what the script does, for when it goes wrong.

---

Every command here needs sudo and is run by the maintainer on the group server.
The app itself never sees the password: nginx terminates TLS and basic auth, and
Streamlit only listens on loopback.

Inbound 443 has been verified reachable from the campus network, so this ends in
a URL. If that ever changes, anyone with SSH gets the identical app through
`ssh -N -L 8501:localhost:8501 user@server` and nothing else about the setup
changes.

## 1. Service account and checkout

```bash
sudo useradd --system --create-home --home-dir /var/lib/fusionui fusionui
sudo mkdir -p /opt/fusion-ui && sudo chown fusionui:fusionui /opt/fusion-ui
sudo -u fusionui git clone https://github.com/Sosnowsky/fusion_ui /opt/fusion-ui
```

### A private dependency the server cannot clone

`experimental_database` is private and the server has no credentials for it.
Copy it in before running the installer — a directory already present in
`$SRC_DIR` is used as it stands and never overwritten:

```bash
rsync -a --exclude .venv --exclude .git ~/Git/experimental_database <host>:/tmp/
ssh <host> "sudo mkdir -p /opt/src && sudo mv /tmp/experimental_database /opt/src/"
```

The installer chowns it to the service user (editable installs write `.egg-info`
into the source tree) and skips the clone. The same applies to any dependency
whose `git pull` fails: the checkout on disk is what gets installed.

The durable fix is a **read-only deploy key**, below. Set that up once and the
installer updates this dependency like every other one.

### Deploy keys for private repositories

Anything private — the app repo itself included — needs a key, because the
service user has no GitHub credentials and `install.sh` never prompts for any
(`GIT_TERMINAL_PROMPT=0`). GitHub has not accepted account passwords for git
since 2021, so a prompt would be a dead end regardless.

**One deploy key grants access to exactly one repository**, and GitHub refuses
to reuse a key across repositories. So repeat this per private repo, with a
different `-f` filename each time. If that becomes more than two or three, use a
machine user or a fine-grained token with read access to all of them instead.

`deploy/add-deploy-key.sh` does all of the below for one repository:

```bash
sudo bash deploy/add-deploy-key.sh https://github.com/uit-cosmo/experimental_database.git
```

It generates the key, wires up the ssh alias and the `insteadOf` rule, prints
the public key with the URL to paste it at, and verifies access afterwards.
Run it once per private repository. The rest of this section is what it does.

The key belongs to the user that does the cloning — the service user, not you.

```bash
# 1. A key for this server, no passphrase (nothing interactive can unlock it)
sudo -u fusionui -H ssh-keygen -t ed25519 -N "" -C "fusion-ui@$(hostname -f)" \
  -f /var/lib/fusionui/.ssh/id_ed25519
sudo -u fusionui -H cat /var/lib/fusionui/.ssh/id_ed25519.pub
```

2. On GitHub: the repository → **Settings → Deploy keys → Add deploy key**.
   Paste the public key, title it after the server, and **leave "Allow write
   access" unchecked**. A deploy key grants access to exactly one repository, and
   GitHub refuses to reuse the same key on a second one — if another dependency
   turns out to be private too, generate a second key for it.

```bash
# 3. Trust github.com, so a clone from cron never waits on a prompt
sudo -u fusionui -H sh -c 'ssh-keyscan github.com >> ~/.ssh/known_hosts'
```

Compare what that recorded against the fingerprints GitHub publishes at
<https://docs.github.com/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints>
before relying on it:

```bash
sudo -u fusionui -H ssh-keygen -lf /var/lib/fusionui/.ssh/known_hosts
```

```bash
# 4. Send just this one repository over SSH, leaving the committed https URL alone
sudo -u fusionui -H git config --global \
  url."git@github.com:uit-cosmo/experimental_database.git".insteadOf \
  "https://github.com/uit-cosmo/experimental_database.git"

# 5. Check it
sudo -u fusionui -H git ls-remote https://github.com/uit-cosmo/experimental_database.git >/dev/null \
  && echo "deploy key works"
```

Step 4 is why `install.sh` needs no edit: the machine-specific credential lives
in the service user's git config, and the URL in the script stays the public
https one. If `/opt/src/experimental_database` was copied in by hand earlier,
delete it so the next run clones properly — or just point its remote at the
SSH URL and leave it.

For a **second** private repo, generate a second key and add its own
`insteadOf` line, plus a `Host` alias so ssh picks the right key:

```bash
sudo -u fusionui -H ssh-keygen -t ed25519 -N "" -f /var/lib/fusionui/.ssh/id_fusion_ui
sudo -u fusionui -H tee -a /var/lib/fusionui/.ssh/config >/dev/null <<'EOF'
Host github-fusion-ui
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_fusion_ui
    IdentitiesOnly yes
EOF
sudo -u fusionui -H git config --global \
  url."github-fusion-ui:Sosnowsky/fusion_ui.git".insteadOf \
  "https://github.com/Sosnowsky/fusion_ui.git"
```

Without `IdentitiesOnly yes` ssh offers every key it has and GitHub answers with
whichever repository the *first accepted* key belongs to — which is how a deploy
key setup ends up cloning the wrong repository and looking haunted.

## 2. Environment

```bash
sudo -u fusionui python3 -m venv /opt/fusion-ui/.venv
# imaging_methods, experimental_database, fusion_scripts, velocity_estimation and
# fppanalysis are not on PyPI — install each editable from its checkout first.
sudo -u fusionui /opt/fusion-ui/.venv/bin/pip install \
  -e /opt/imaging-methods -e /opt/experimental_database -e /opt/fusion_scripts \
  -e /opt/velocity-estimation -e /opt/fpp-analysis-tools
sudo -u fusionui /opt/fusion-ui/.venv/bin/pip install -e /opt/fusion-ui

sudo -u fusionui cp /opt/fusion-ui/.env.example /opt/fusion-ui/.env
sudo -u fusionui $EDITOR /opt/fusion-ui/.env      # paths for this server
```

The two writable locations are `FUSION_UI_DB` and `FUSION_UI_CACHE`; everything
else is read-only to the service.

```bash
sudo mkdir -p /hdd1/fusion_ui/cache
sudo chown -R fusionui:fusionui /hdd1/fusion_ui
```

## 3. Database and first index

```bash
sudo -u fusionui /opt/fusion-ui/.venv/bin/fusion-ui init-db
sudo -u fusionui /opt/fusion-ui/.venv/bin/fusion-ui rescan
sudo -u fusionui /opt/fusion-ui/.venv/bin/fusion-ui status
```

`status` is the check that the paths in `.env` resolve and the index is filled.

Keep the index fresh — new shots appear in the browser only after a rescan:

```bash
sudo -u fusionui crontab -e
# every 15 minutes
*/15 * * * * /opt/fusion-ui/.venv/bin/fusion-ui rescan >> /var/log/fusion-ui-rescan.log 2>&1
```

## 4. systemd

```bash
sudo cp /opt/fusion-ui/deploy/fusion-ui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fusion-ui
sudo systemctl status fusion-ui
curl -sI http://127.0.0.1:8501/ | head -1        # expect 200
```

`ReadWritePaths` in the unit is set to `/hdd1/fusion_ui` — change it if `.env`
puts the database or cache somewhere else, or the service will fail to write.

## 5. nginx, TLS and the shared password

```bash
sudo apt install nginx apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd fusion    # one shared account
sudo mkdir -p /etc/nginx/ssl
sudo openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/fusion-ui.key -out /etc/nginx/ssl/fusion-ui.crt \
  -subj "/CN=$(hostname -f)"

sudo cp /opt/fusion-ui/deploy/nginx.conf /etc/nginx/sites-available/fusion-ui
sudo ln -sf /etc/nginx/sites-available/fusion-ui /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

## 6. Firewall

```bash
ip -4 route show default                 # which interface faces the network
ip -4 -o addr show dev <iface>           # its address and prefix
sudo ufw allow from <subnet> to any port 443 proto tcp
sudo ufw status
```

The server's own subnet is the right answer in almost every case — the laptops
are on the same network. `install.sh` works it out and offers it, but never
applies it without you saying so: the interface prefix can be wider than you
meant, and a firewall rule is about who gets in.

The failure seen while verifying reachability was the *host* firewall rejecting
with ICMP host-unreachable — an immediate "No route to host", not a timeout.
A timeout instead means the campus firewall; those are different tickets.

## 7. Verify — including the websocket

```bash
bash deploy/verify.sh <host>
```

That runs the four checks below. The last one is the one that matters: a plain
`curl` proving the HTML arrives is **not** enough, because if the proxy eats the
`Upgrade` header the page loads and then hangs forever with no error message.

```bash
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8501/   # 200 — streamlit is up
curl -sk -o /dev/null -w '%{http_code}' https://<host>/         # 401 — password enforced
curl -sk -o /dev/null -w '%{http_code}' -u fusion:<password> https://<host>/   # 200

curl -k -u fusion:<password> -i -N \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: $(openssl rand -base64 16)" \
  https://<host>/_stcore/stream
```

Expect `HTTP/1.1 101 Switching Protocols`. A `200` means the upgrade was
swallowed — the app will look broken in the browser for everyone.

Then, from a laptop on the campus network, open `https://<host>/`, log in, and
confirm the shot browser lists shots.

## Routine operations

```bash
sudo systemctl restart fusion-ui                  # after a git pull
sudo journalctl -u fusion-ui -f                   # logs
sudo -u fusionui /opt/fusion-ui/.venv/bin/fusion-ui status
```

The whole of the app's state is the SQLite file and the cache directory named in
`.env`. Back up the SQLite file; the cache can always be recomputed.
