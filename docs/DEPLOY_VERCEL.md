# Frontend on Vercel, backend in Docker on your PC

Vercel serves the static React app; the API, model, Celery worker and Redis keep running in Docker on this PC and are exposed through a Cloudflare tunnel (`cloudflared`). Hosting cost is zero, but the site only works while the PC is on, Docker is running and the tunnel is up.

## Choose a tunnel mode

| | Named tunnel (`--profile public`) | Quick tunnel (`--profile quick`) |
|---|---|---|
| URL | stable, e.g. `https://api.yourdomain.com` | random `*.trycloudflare.com`, changes each restart |
| Needs | free Cloudflare account + a domain on Cloudflare | nothing |
| Vercel redeploy | once | after every tunnel restart |

### Named tunnel setup (once)
1. Cloudflare dashboard → Zero Trust → Networks → Tunnels → **Create a tunnel** (Cloudflared). Copy the **token**.
2. Add a **public hostname** (e.g. `api.yourdomain.com`) with service `HTTP` → `backend:8000`.
3. Put the token in the root `.env` (gitignored, never commit it): `CLOUDFLARE_TUNNEL_TOKEN=...`

## Vercel (once, from `src/frontend`)
```
npm i -g vercel
vercel login
vercel --prod --build-env VITE_API_URL=https://api.yourdomain.com
```
Or in the dashboard: import the repo, **Root Directory** `src/frontend`, env var `VITE_API_URL=https://api.yourdomain.com` (no trailing slash). Note the production URL (e.g. `https://your-app.vercel.app`).

## Allow that origin and start
Add to the root `.env`: `VERCEL_ORIGIN=https://your-app.vercel.app`, then
```
docker compose --profile public up -d     # or: --profile quick
```
Check `https://api.yourdomain.com/health` shows `model_loaded: true`, then open the Vercel URL. For quick mode, read the URL with `docker compose logs tunnel-quick | grep trycloudflare`, set it as `VITE_API_URL` in Vercel and redeploy.

## Day-to-day
Start `docker compose --profile public up -d`; stop `docker compose --profile public down`.

## Caveats
- The API has no authentication, only per-IP rate limits (e.g. `/api/batch`: 5 requests / 5 min). Anyone with the URL can use your CPU.
- CPU inference: a second or two per request; SHAP explanations are slower.
- Vercel preview deployments have other URLs and are blocked by CORS unless added to `FRONTEND_ORIGIN`.
