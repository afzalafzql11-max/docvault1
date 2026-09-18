# DocuVault AI — Streamlit Document Vault

A DigiLocker-style college/final-year-project prototype built with Streamlit.

## Features

- Sign up and login
- Password hashing with PBKDF2
- Upload PDF/images
- AI document scanning with Gemini API (Google GenAI SDK)
- Extract document name/type/holder name/issue date/expiry date
- Translate important document text into English
- Save document metadata
- Download original uploaded document
- Dashboard with validity counts
- Expiry warnings (expired / <= 30 days)
- Calendar-style expiry table
- Works without Gemini using a limited local OCR fallback for images
- Render Free compatible startup command

## Important Render Free limitation

Render Free web services have an ephemeral filesystem. SQLite databases and uploaded
files stored on the service can be lost when the service restarts, redeploys, or spins down.

Therefore this repository is a **prototype/demo**, not a production document vault.

For a real deployment, replace the local SQLite/file storage with persistent services
such as Supabase Postgres + Supabase Storage (or another external database/object store).

## Gemini API (Google GenAI SDK)

Gemini has a free tier for supported models, but quotas/rate limits apply. Do not put
your API key in GitHub.

1. Create a Gemini API (Google GenAI SDK) key in Google AI Studio.
2. Push this project to GitHub.
3. In Render, open the service's Environment Variables.
4. Add:
   GEMINI_API_KEY = your key
5. Deploy/redeploy.

The application is intentionally designed so that if Gemini is missing or fails, it does
not crash. It falls back to limited local OCR for image uploads.

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## Render

The included render.yaml can be used as the service blueprint.

Build:
```text
pip install -r requirements.txt
```

Start:
```text
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

## Security note

This is an academic prototype. Do not upload real identity documents to a public demo
until you have implemented production-grade encryption, persistent private object
storage, access controls, audit logging, secure session management, retention/deletion
policies, and applicable privacy/legal requirements.

\n## PDF extraction fix
The app uses Google's current `google-genai` SDK and the Gemini Files API for
document uploads. PDFs are uploaded to Gemini as actual PDF documents before
the extraction request, rather than being passed as a generic byte dictionary.
