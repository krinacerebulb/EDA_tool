# Auto EDA — Server Deploy

Self-contained folder for deploying the Auto EDA Streamlit app.
Upload **this folder** as-is to your server / hosting provider.

## What's inside

```
deploy/
├── app.py                  # Streamlit entry point
├── modules/                # business logic
├── utils/                  # helpers
├── requirements.txt        # Python deps
├── .streamlit/config.toml  # server-safe Streamlit settings
├── Procfile                # Render / Railway / Heroku start command
├── Dockerfile              # for any container host
├── .dockerignore
└── .gitignore
```

Nothing else from the dev repo is required at runtime.

---

## Option 1 — Streamlit Community Cloud (free, easiest)

1. Push this folder to a GitHub repo.
2. Go to <https://share.streamlit.io> → **New app**.
3. Pick the repo, set **Main file path** to `app.py`.
4. Deploy. Done.

---

## Option 2 — Render / Railway / Fly.io / Heroku

The included `Procfile` already has the right start command.

**Render:**
1. New → Web Service → connect repo.
2. Environment: **Python 3**.
3. Build command: `pip install -r requirements.txt`
4. Start command: leave empty (uses Procfile).

**Railway:** New project → Deploy from GitHub → it picks up the Procfile automatically.

---

## Option 3 — Docker (any container host: AWS, GCP, Azure, your own)

```bash
docker build -t auto-eda .
docker run -p 8501:8501 auto-eda
```

Open <http://localhost:8501>. To deploy, push the image to your registry of choice.

---

## Option 4 — Plain Linux VPS (Ubuntu / Debian)

```bash
# 1. Upload the folder, then on the server:
cd /opt/auto-eda
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run (foreground test)
streamlit run app.py --server.port=8501 --server.address=0.0.0.0

# 3. For production, run as a systemd service. Create
#    /etc/systemd/system/auto-eda.service:
```

```ini
[Unit]
Description=Auto EDA Streamlit
After=network.target

[Service]
WorkingDirectory=/opt/auto-eda
ExecStart=/opt/auto-eda/.venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
User=www-data

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now auto-eda
sudo systemctl status auto-eda
```

Then put nginx (or Caddy) in front for HTTPS / domain.

---

## Local sanity check before uploading

```bash
cd deploy
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS
pip install -r requirements.txt
streamlit run app.py
```

If it opens at <http://localhost:8501> and works, the server deploy will work too.
