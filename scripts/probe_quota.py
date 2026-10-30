"""Check whether the Gemini judge and answer models have quota right now (prints no key)."""
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
for m in ["gemini-3.6-flash", "gemini-3.5-flash"]:
    try:
        r = c.models.generate_content(model=m, contents="Reply with OK")
        print(m, "-> available:", (r.text or "").strip()[:20])
    except Exception as e:
        s = str(e)
        kind = "daily quota" if "PerDay" in s else "per-minute quota" if "PerMinute" in s else ""
        print(m, "->", s[:70], "|", kind)
