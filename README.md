# EffectLab — AI Video Effects Studio

A Next.js starter for a creator-facing photo-to-video AI effects product with a dark dashboard, effect presets, token-cost UI, monthly plan cards, an image-to-video API route, and Stripe Checkout scaffolding.

## Current status

- Responsive product dashboard and effects gallery are implemented.
- Photo upload/preview is implemented.
- Real image-to-video generation is wired to the FAL client and Kling image-to-video model. Add `FAL_KEY` to enable it; provider usage is billed by the provider.
- Stripe subscription Checkout route is scaffolded for three monthly plans. Add Stripe recurring Price IDs and `STRIPE_SECRET_KEY` to enable checkout.
- **This is a starter, not yet safe to sell as a live paid product.** The token counter is currently demo-only and browser-side. There is no account authentication, database-backed token ledger, Stripe webhook handling, job persistence, rate limiting, or refund logic. Do not charge users for token allowances until those are implemented. Checkout alone does not activate credits.

## Run locally

Requirements: Node.js 20+.

```bash
npm install
cp .env.example .env.local
npm run dev
```

Open http://localhost:3000.

## Enable AI generation

1. Create a FAL account and API key.
2. Set `FAL_KEY` in `.env.local` (never put it in client-side code or commit the real value).
3. Run `npm run dev` and test an image upload.

The `/api/generate` endpoint uploads the image to provider storage, submits a vertical 5-second image-to-video job, and returns the generated video URL. Provider/model availability, pricing, queue times and output quality depend on your FAL account and the selected model.

## Configure Stripe Checkout

1. Create three recurring monthly prices in Stripe: £7.99, £17.99, and £39.99.
2. Add the corresponding Price IDs as `STRIPE_PRICE_STARTER`, `STRIPE_PRICE_CREATOR`, and `STRIPE_PRICE_STUDIO`.
3. Add `STRIPE_SECRET_KEY` (start with a test key).

The checkout route redirects customers to Stripe, but it does not yet grant tokens. Before public launch, implement Supabase (or another trusted database) authentication and a server-side token ledger, then process verified Stripe webhooks for subscription creation, renewal, cancellation and payment failure. Never trust token balances sent from a browser.

## Deploy

Recommended: import this private GitHub repository into Vercel, set the environment variables in the Vercel project settings, and deploy. GitHub Pages cannot run the Next.js API routes or safely store API keys, so it is not suitable for the complete app.

## Project layout

- `app/page.jsx` — dashboard, upload flow, effect presets, plans
- `app/globals.css` — responsive styling
- `app/api/generate/route.js` — FAL image-to-video integration
- `app/api/checkout/route.js` — Stripe subscription Checkout session
- `.env.example` — names of required environment variables (no secrets)

## Responsible use

Only upload images you have rights to use. Add an abuse policy, content safeguards, deletion controls and privacy/retention disclosures before accepting public uploads.
