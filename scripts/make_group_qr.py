"""Regenerate the WhatsApp-group QR code used on the pairings doc.

One-off helper: the rendered PNG is committed as
``tmp/westside_whatsapp_group_qr.png`` so ``pairings_docx`` needs no QR
library at runtime (nothing extra to install on the Pi). Re-run this
only if the group invite link changes:

    pip install qrcode pillow
    python scripts/make_group_qr.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qrcode
from qrcode.constants import ERROR_CORRECT_M

from pairings_docx import JOIN_GROUP_QR_PATH, JOIN_GROUP_URL


def main() -> None:
    qr = qrcode.QRCode(error_correction=ERROR_CORRECT_M, box_size=10, border=2)
    qr.add_data(JOIN_GROUP_URL)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(str(JOIN_GROUP_QR_PATH))
    print(f"wrote {JOIN_GROUP_QR_PATH} (version {qr.version}, {img.size[0]}px)")


if __name__ == "__main__":
    main()
