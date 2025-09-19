import os, ssl, smtplib
from email.mime.text import MIMEText
from email.utils import formatdate
from dotenv import load_dotenv
import requests

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../config/.env"))

SMTP_SERVER   = os.getenv("SMTP_SERVER", "")
SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER     = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM    = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO      = [x.strip() for x in os.getenv("EMAIL_TO", "").split(",") if x.strip()]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

def send_email(subject: str, body: str, to_addrs=None) -> bool:
    to_addrs = to_addrs or EMAIL_TO
    if not (SMTP_SERVER and SMTP_USER and SMTP_PASSWORD and to_addrs):
        print("[email] Missing SMTP config or recipients.")
        return False
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(to_addrs)
    msg["Date"] = formatdate(localtime=True)

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(EMAIL_FROM, to_addrs, msg.as_string())
        return True
    except Exception as e:
        print(f"[email] error: {e}")
        return False

def send_telegram(text: str) -> bool:
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        print("[tg] Missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=10)
        if r.ok:
            return True
        print(f"[tg] error: {r.status_code} {r.text}")
        return False
    except Exception as e:
        print(f"[tg] error: {e}")
        return False
