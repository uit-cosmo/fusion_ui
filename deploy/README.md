# Deploying Shot Explorer

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
sudo ufw allow from <campus-subnet> to any port 443 proto tcp
sudo ufw status
```

The failure seen while verifying reachability was the *host* firewall rejecting
with ICMP host-unreachable — an immediate "No route to host", not a timeout.
A timeout instead means the campus firewall; those are different tickets.

## 7. Verify — including the websocket

A plain `curl` proving the HTML arrives is **not** enough. If the proxy eats the
`Upgrade` header, the page loads and then hangs forever with no error message.
Test the websocket handshake explicitly:

```bash
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
