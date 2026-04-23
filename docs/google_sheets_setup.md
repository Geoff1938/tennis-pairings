# Google Sheets setup for Boris

Boris reads (and will write) the player roster from a Google Sheet. Admins
edit the sheet directly when they want to update ratings, add notes, etc.
This is one-off setup — once done the JSON key lives in `.env` and no
further auth is needed.

**Cost**: £0. The Google Sheets API is free for personal use; Cloud
project creation is free; storage counts against your 15 GB Drive quota.

**Time**: ~10 min of clicking. None of this is scary but the Google Cloud
console is a bit busy.

---

## 1. Create a Google Cloud project

1. Open https://console.cloud.google.com/
2. Sign in with the Google account that owns (or will own) the sheet
   (your personal Google account is usually the easiest choice).
3. Top-left, next to the "Google Cloud" logo, click the project picker →
   **NEW PROJECT**.
4. Name it something like **`tennis-pairings`**. Leave Organization as "No
   organization". Click **CREATE**. Wait ~10 s for it to finish.
5. Make sure the new project is the **selected** one in the project
   picker before continuing.

## 2. Enable the Sheets API

1. In the left nav (hamburger menu), go to **APIs & Services → Library**.
2. Search for **"Google Sheets API"**. Click it.
3. Click **ENABLE**.
4. (Optional but useful) do the same for **"Google Drive API"** — lets the
   service account list sheets by name instead of hard-coding sheet IDs.

## 3. Create a service account

1. **APIs & Services → Credentials → + CREATE CREDENTIALS → Service
   account**.
2. Name it **`tennis-bot`** (or whatever). Click **CREATE AND CONTINUE**.
3. Role: skip (leave blank) → **CONTINUE** → **DONE**.
4. You now see a list of service accounts. Click the one you just created
   to open it.
5. Top tab **KEYS** → **ADD KEY → Create new key → JSON → CREATE**. A
   `.json` file downloads.
6. **Rename the file to `gcp_service_account.json`** and **move it into
   the project root** (`C:\Users\gicha\GC-repos\tennis-pairings\`). It's
   already in `.gitignore`.
7. **Copy the service account's email** (looks like
   `tennis-bot@tennis-pairings.iam.gserviceaccount.com`). You'll need it
   in step 5.

## 4. Create the Google Sheet

1. Open https://sheets.google.com/ and create a new sheet.
2. Rename it **`Tennis Pairings`**.
3. Create three tabs (right-click the tab at the bottom → Rename /
   Duplicate, etc.):
   - **Players** — roster
   - **Session Log** — one row per session (date, attendees, courts,
     rotations, notes)
   - **Pair Log** — one row per rotation-court-pair for browsing
4. In the **Players** tab, set row 1 to:
   `name | gender | rating | notes`
5. Leave the other tabs empty — Boris will write headers on first use.

## 5. Share the sheet with the service account

1. In the sheet, click **Share** (top right).
2. Paste the service account email (from step 3.7) — e.g.
   `tennis-bot@tennis-pairings.iam.gserviceaccount.com`.
3. Give it **Editor** access.
4. Uncheck "Notify people" (the service account can't read email).
5. Click **Share**.

## 6. Record the sheet ID

1. Open the sheet in the browser. The URL looks like:
   `https://docs.google.com/spreadsheets/d/AbCdEfGhIjKlMnOp0123456789ABCDEF/edit#gid=0`
2. Copy the long string between `/d/` and `/edit` — that's the
   **sheet ID**.
3. Add it to your `.env` file:
   ```
   GOOGLE_SHEET_ID=AbCdEfGhIjKlMnOp0123456789ABCDEF
   ```

## 7. Tell me when it's done

When all seven steps are complete, tell me and I'll:
- Add `gspread` to the Python deps
- Swap `roster.py`'s backend from local JSON → Google Sheets (behind the
  same public API, so none of the bot tools change)
- Migrate the current 41 players from `players.json` into the **Players**
  tab
- Wire auto-add-on-new-player and set-rating through to the sheet
- Append session/pair rows to the **Session Log** / **Pair Log** tabs
  after every run

Until then, Boris continues to use `players.json` locally — no behaviour
change visible to you.

## Troubleshooting

- **"API has not been used in project"**: you forgot step 2 (enable
  Sheets API). Enable it, wait 30 s, try again.
- **"The caller does not have permission"**: the service account isn't
  shared on the sheet (step 5), or you shared with the wrong email.
  Double-check the email.
- **"Requested entity was not found"**: the `GOOGLE_SHEET_ID` in `.env`
  is wrong. Copy-paste carefully from the URL.
- **Credentials JSON not found**: the file must be at
  `tennis-pairings/gcp_service_account.json` (exact name, project root).
