# GestureFX Web

Browser-based version of the GestureFX hand gesture visual effects app.
Uses the webcam + MediaPipe.js for real-time hand tracking, rendered on Canvas
with neon skeleton and gesture-triggered effects.

## Features

- 🖐️ Real-time hand skeleton (neon glow)
- 🛡️ Gesture effects: shield (open palm), beam (point), particles (fist), trail (move)

## Deploy to Vercel

The repo root already contains a `vercel.json` with
`"outputDirectory": "web-app"`, so the site deploys with **zero dashboard
configuration** — just connect the GitHub repo and hit Deploy.

1. Go to [vercel.com/new](https://vercel.com/new).
2. Import the `wbsanjar/ai` GitHub repo.
3. Click **Deploy**. That's it — the root `vercel.json` tells Vercel to serve
   the static `web-app/` folder.

Alternative via CLI:

```bash
cd <repo-root>
npm i -g vercel
vercel --prod
```

This is a fully static site — no backend needed.

> Note: Your browser must have camera permission and the page must be served
> over HTTPS (Vercel gives you this automatically) for the webcam to work.

## Run locally

Serve the folder over HTTP (camera needs a secure context; use `localhost`):

```bash
npx serve .
```

Then open `http://localhost:3000`.
