# Deploying Shot Explorer

Streamlit listens on loopback, run by systemd. nginx in front terminates TLS and
checks one shared password, so the app never sees it. Every command here needs
sudo and is run by the maintainer on the group server.

```bash
git clone https://github.com/uit-cosmo/fusion_ui.git ~/fusion_ui
sudo bash ~/fusion_ui/deploy/install.sh
```

`install.sh` is every step below, in order, and each step is idempotent: run it
again to update a deployment in place. It stops three times for you: to edit
`.env` (step 3), to set the shared password (step 7) and to confirm the firewall
rule (step 8).

`--help` lists the settings: `--branch` (default `main`; the script checks that
branch out in the app's checkout), `--repo-url`, `--app-dir`, `--src-dir`,
`--state-dir`, `--user`, `--server-name`, `--subnet`, `--htpasswd-user`. Each
also reads the matching environment variable, but prefer the flags: whether
`sudo VAR=value bash …` passes a variable through depends on the sudoers config.

The current deployment is **<https://fp1-hpz4fusion.int.uit.no/>**: that name,
not the machine's own `fusion-HP-Z4-G4-Workstation`, which resolves nowhere.
Port 80 is closed there, so the `https://` has to be typed. The certificate is
self-signed, so the first visit needs one click through the browser's warning.

If inbound 443 is ever blocked, anyone with SSH gets the same app through
`ssh -N -L 8501:localhost:8501 user@server`, and nothing else changes.

To check a deployment at any time:

```bash
bash deploy/verify.sh <host>
```

**Before running `install.sh` again on a server where another service holds
port 80, as apache does on the group server, read "Port 80" under step 7.**

The rest of this file is what the script does, step by step, for when it goes
wrong. `$APP_DIR` is the checkout the service runs from (the checkout
`install.sh` is run from, unless `--app-dir`), `$SRC_DIR` is where the
dependency checkouts live (the checkout's parent, unless `--src-dir`), and
`$STATE_DIR` holds the SQLite file and the result cache (`/hdd1/fusion_ui`,
unless `--state-dir`).

## 1. Packages and service account

```bash
sudo apt-get install -y --no-install-recommends \
  git python3-venv nginx apache2-utils openssl curl
sudo useradd --system --create-home --home-dir /var/lib/fusionui fusionui
```

## 2. Checkouts

The script pulls or clones four dependencies into `$SRC_DIR`, then the app into
`$APP_DIR`:

| Checkout | From | |
|---|---|---|
| `imaging-methods` | `uit-cosmo/imaging-methods` | public |
| `velocity-estimation` | `uit-cosmo/velocity-estimation` | public |
| `fpp-analysis-tools` | `uit-cosmo/fpp-analysis-tools` | public |
| `experimental_database` | `uit-cosmo/experimental_database` | **private** |

For each one:

- A git checkout already there is updated with `git pull --ff-only`. If that
  fails (no credentials, a detached HEAD, local edits), the checkout is used as
  it stands.
- A directory already there that is not a git checkout is used as it stands and
  never overwritten.
- Otherwise it is cloned. A clone that fails stops the install.

Each tree is then shared with the service user through the group, not handed
over, so a checkout you own stays yours:

```bash
sudo chgrp -R fusionui "$APP_DIR" && sudo chmod -R g+rwX "$APP_DIR"
sudo find "$APP_DIR" -type d -exec chmod g+s {} +
```

It is `g+rwX`, not `g+w`: group-write alone leaves `drwx------` directories
untraversable, and pip dies listing them. The service user writes there because
an editable install writes `.egg-info` into the source tree and `git pull`
writes `.git`.

`$SRC_DIR` itself gets `chgrp fusionui` and `chmod g+xs,g-w`: the service must
traverse it and never write it. It is usually your home directory, and sshd
ignores `~/.ssh/authorized_keys` in a group-writable home.

The checkouts are installed editable, so a `git pull` in one of them changes the
next `fusion-ui` command at once, and the service at its next restart. Checkouts
the script cloned belong to the service user; the `usermod` of step 3 lets you
pull in them.

### A private repository the server cannot clone

Either give the service user a deploy key (below), or copy the checkout in
before running the installer:

```bash
rsync -a --exclude .venv --exclude .git ~/Git/experimental_database <host>:/tmp/
ssh <host> "sudo mkdir -p $SRC_DIR && sudo mv /tmp/experimental_database $SRC_DIR/"
```

A copy is installed as it is and is never updated by the script. With a deploy
key the script updates the dependency like the others; delete a copy made
earlier, so that the next run clones it.

### Deploy keys

`install.sh` never prompts for credentials (`GIT_TERMINAL_PROMPT=0`), and GitHub
does not accept account passwords for git. **One deploy key gives access to
exactly one repository**, and GitHub refuses the same key on a second one, so
run this once per private repository:

```bash
sudo bash deploy/add-deploy-key.sh https://github.com/uit-cosmo/experimental_database.git
```

For `<owner>/<repo>` it generates `/var/lib/fusionui/.ssh/id_<repo>` for the
service user (who does the cloning; no passphrase, since nothing interactive
can unlock it), adds an ssh alias `Host github-<repo>` that uses that key with
`IdentitiesOnly yes`, and sets a git `insteadOf` rule that sends the
repository's https URL through the alias. So the URLs in `install.sh` stay the
public https ones. Then it prints the public key and the page to paste it at
(the repository's **Settings → Deploy keys**), waits, and checks access.

- **Leave "Allow write access" unchecked.**
- Compare the github.com host-key fingerprints it prints with the ones GitHub
  publishes at
  <https://docs.github.com/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints>.
- If you set a key up by hand, use the same alias form. Without
  `IdentitiesOnly yes` ssh offers every key it has, and GitHub answers for
  whichever repository the first accepted key belongs to, which is how a setup
  ends up cloning the wrong repository. A rewrite to a bare
  `git@github.com:<owner>/<repo>.git` uses the default identity, works only
  while there is one key, and shadows the alias rule (the script removes it).
- Past three or four private repositories, use a machine user or a fine-grained
  token with read access to all of them.

### A public repository asks for a username

Then the problem is not access. The cause seen on this server is a git bug:
**git 2.43 built against a GnuTLS curl gets a spurious `401 Basic
realm="GitHub"` on the protocol-v2 `POST /git-upload-pack`**, because it reuses
the HTTP/2 connection left from the ref advertisement. An anonymous clone of a
public repository turns into a credential prompt. `install.sh` pins
`http.version` to HTTP/1.1 for its own git calls to avoid it. To confirm:

```bash
sudo -u fusionui -H env GIT_TERMINAL_PROMPT=0 \
  git ls-remote https://github.com/uit-cosmo/velocity-estimation.git   # 401
sudo -u fusionui -H env GIT_TERMINAL_PROMPT=0 \
  git -c http.version=HTTP/1.1 \
  ls-remote https://github.com/uit-cosmo/velocity-estimation.git       # works
```

If the second fails too, it is configuration. Look for an `insteadOf` line that
catches more than its own repository:

```bash
sudo -u fusionui -H git config --list --show-origin | grep -iE 'url\.|insteadof|credential'
```

`sudo` does not forward the environment: `GIT_TERMINAL_PROMPT=0 sudo -u fusionui
git …` loses the variable without a word. Every git call in these scripts
therefore goes through `sudo -u … env VAR=… git`.

## 3. Configuration

Two accounts write the state directory: the service, and whoever runs
`fusion-ui precompute` or `rescan` by hand. Make it group-writable before
anything is created in it:

```bash
sudo mkdir -p "$STATE_DIR/cache"
sudo chown -R fusionui:fusionui "$STATE_DIR"
sudo chmod -R g+w "$STATE_DIR"
sudo find "$STATE_DIR" -type d -exec chmod g+s {} +   # new entries inherit the group
sudo usermod -aG fusionui <you>                       # log out and back in
```

The app creates its own directories and blobs group-writable
(`fusion_ui/core/shared.py`), but it cannot choose the group: the `chown` and
the setgid bit are the operator's half. Without them a directory made by one
account is unwritable by the other, which costs the second one a whole analysis
before it finds out. The script runs these on every run, which also repairs a
tree created before the app did its half.

If `$APP_DIR/.env` does not exist, the script copies `.env.example`, sets
`FUSION_UI_DB=$STATE_DIR/shot_explorer.sqlite` and
`FUSION_UI_CACHE=$STATE_DIR/cache`, and opens it in `$EDITOR` (nano by default)
for you to set `FUSION_DISCHARGE_DB` and `FUSION_DATA_FOLDER`. An existing `.env`
is left alone. The database and the cache are the only places the service
writes.

## 4. Python environment

```bash
sudo -u fusionui python3 -m venv "$APP_DIR/.venv"
sudo -u fusionui "$APP_DIR/.venv/bin/pip" install --upgrade pip
# The checkouts first: the app names them as dependencies, and not all are on PyPI.
sudo -u fusionui "$APP_DIR/.venv/bin/pip" install \
  -e "$SRC_DIR/imaging-methods" -e "$SRC_DIR/experimental_database" \
  -e "$SRC_DIR/velocity-estimation" -e "$SRC_DIR/fpp-analysis-tools"
sudo -u fusionui "$APP_DIR/.venv/bin/pip" install -e "$APP_DIR"
```

## 5. Database and first index

```bash
sudo -u fusionui "$APP_DIR/.venv/bin/fusion-ui" init-db
sudo -u fusionui "$APP_DIR/.venv/bin/fusion-ui" rescan
sudo -u fusionui "$APP_DIR/.venv/bin/fusion-ui" status
```

`status` is the check that the paths in `.env` resolve and the index is filled.

New shots appear in the browser only after a rescan, so the script adds this
line to the service user's crontab, once, and creates the log file for it:

```
*/15 * * * * $APP_DIR/.venv/bin/fusion-ui rescan >> /var/log/fusion-ui-rescan.log 2>&1
```

## 6. systemd

The unit file is a template: the script fills in `@APP_DIR@`, the user and
group, and `ReadWritePaths`. Copying it unchanged does not work.

```bash
sudo sed -e "s|@APP_DIR@|$APP_DIR|g" \
    -e "s|^ReadWritePaths=.*|ReadWritePaths=$STATE_DIR|" \
    "$APP_DIR/deploy/fusion-ui.service" | sudo tee /etc/systemd/system/fusion-ui.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now fusion-ui
sudo systemctl status fusion-ui
curl -sI http://127.0.0.1:8501/ | head -1        # expect 200
```

`ReadWritePaths` is `$STATE_DIR`. If `.env` puts the database or the cache
somewhere else, change it, or the service fails to write.

## 7. nginx, TLS and the shared password

```bash
sudo htpasswd -c /etc/nginx/.htpasswd fusion    # one shared account
sudo mkdir -p /etc/nginx/ssl
ip=$(hostname -I | awk '{print $1}')
name=$(getent hosts "$ip" | awk '{print $2}')
sudo openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/fusion-ui.key -out /etc/nginx/ssl/fusion-ui.crt \
  -subj "/CN=$name" -addext "subjectAltName=DNS:$name,IP:$ip"
```

The script creates the password file only if there is none; to change the
password, run `sudo htpasswd /etc/nginx/.htpasswd fusion`.

Two things about the name in the certificate:

- It must be the name **DNS** knows, not `hostname -f`. This server calls itself
  `fusion-HP-Z4-G4-Workstation` and answers campus-wide as
  `fp1-hpz4fusion.int.uit.no`; a certificate for the first is a warning plus a
  URL nobody can open. The script takes the reverse lookup of the server's
  address, when it resolves forward to the same address, and `hostname -f`
  otherwise. `--server-name` overrides.
- It must be in `subjectAltName`, not only in the subject: browsers ignore CN.

The script reissues the certificate whenever the existing one does not cover the
server name, so correcting `--server-name` and running it again is enough.

```bash
sudo sed -e "s|server_name _;|server_name $name;|" "$APP_DIR/deploy/nginx.conf" |
  sudo tee /etc/nginx/sites-available/fusion-ui >/dev/null
sudo ln -sf /etc/nginx/sites-available/fusion-ui /etc/nginx/sites-enabled/fusion-ui
sudo rm -f /etc/nginx/sites-enabled/default
sudo mkdir -p /etc/systemd/system/nginx.service.d
sudo cp "$APP_DIR/deploy/nginx-restart.conf" \
  /etc/systemd/system/nginx.service.d/fusion-ui-restart.conf
sudo systemctl daemon-reload
sudo nginx -t && sudo systemctl enable --now nginx && sudo systemctl reload nginx
```

Besides the server name, the script makes two more changes to `nginx.conf` on
the way: on nginx 1.25.1 or later it replaces `listen 443 ssl http2;` by
`listen 443 ssl;` and `http2 on;`, and it may remove the http redirect (next
section).

The drop-in is there because Ubuntu's nginx unit never restarts a failed start,
and an unattended upgrade restarts nginx. With it nginx retries every 30 s for
as long as it fails, so a conflict that clears up brings the site back by
itself.

### Port 80

`nginx.conf` has a block that redirects http to https, between the lines
`# BEGIN http-redirect` and `# END http-redirect`. nginx binds every `listen`
at startup, so if another service holds port 80 (apache, on this server), that
block stops nginx from starting at all, and takes the 443 site down with it:

```
bind() to 0.0.0.0:80 failed (98: Address already in use)      # journalctl -u nginx
```

This has happened: an unattended upgrade started apache again, the next nginx
restart failed, and the site was down for six days. The deployed site config has
no redirect block and serves 443 only.

**The script removes the block only when port 80 is taken and nginx is not
running.** Run on a server where nginx is already up, it writes the block back
into `/etc/nginx/sites-available/fusion-ui`. So after every run of `install.sh`
on a server where another service holds port 80:

```bash
grep -n 'listen 80' /etc/nginx/sites-available/fusion-ui    # expect no output
```

If it prints a line, delete the `server { listen 80; … }` block from that file,
then `sudo nginx -t && sudo systemctl reload nginx`.

To take port 80 back for the redirect, stop the other service and run the
installer again.

## 8. Firewall

```bash
ip -4 route show default                 # which interface faces the network
ip -4 -o addr show dev <iface>           # its address and prefix
sudo ufw allow from <subnet> to any port 443 proto tcp
sudo ufw status
```

The server's own subnet is the right answer in almost every case, since the
laptops are on the same network. The script works it out and offers it, but
never applies it without you saying so: the interface prefix can be wider than
you meant. If a rule for 443 exists and no `--subnet` is given, it leaves the
firewall alone. If ufw is inactive, the rule filters nothing; `sudo ufw allow
OpenSSH && sudo ufw enable` turns it on without locking you out of ssh.

An immediate "No route to host" from a laptop is the *host* firewall rejecting
the connection. A timeout is the campus firewall. Those are different tickets.

## 9. Verify

```bash
bash deploy/verify.sh <host> [user]
```

It asks for the shared password and runs four checks. From a laptop the first
one fails, since it asks the laptop's own port 8501; the other three tell.

| Check | Expect |
|---|---|
| Streamlit on loopback, `http://127.0.0.1:8501/` | 200 |
| `https://<host>/` with the password | 200 |
| `https://<host>/` without it | 401 |
| The websocket handshake on `https://<host>/_stcore/stream` | 101 |

The last is the one that matters. Streamlit talks over a websocket, and a proxy
that drops the `Upgrade` header serves a page that loads and then hangs forever
with no error. A 200 on the page proves nothing. By hand:

```bash
curl -k -u fusion:<password> -i -N --http1.1 \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" \
  -H "Sec-WebSocket-Key: $(openssl rand -base64 16)" \
  https://<host>/_stcore/stream
```

Expect `HTTP/1.1 101 Switching Protocols`. A `200` means nginx swallowed the
upgrade: check the `proxy_set_header Upgrade` and `Connection` lines of the
site config. Keep `--http1.1`: HTTP/2 cannot carry an `Upgrade`, so without it
curl reports a 200 that means nothing.

Then, from a laptop on the campus network, open `https://<host>/`, log in, and
confirm that the shot browser lists shots.

## Routine operations

```bash
sudo systemctl restart fusion-ui                  # after a git pull
sudo journalctl -u fusion-ui -f                   # logs
sudo -u fusionui "$APP_DIR/.venv/bin/fusion-ui" status
```

All of the app's state is the SQLite file and the cache directory named in
`.env`. Back up the SQLite file; the cache can be recomputed.
