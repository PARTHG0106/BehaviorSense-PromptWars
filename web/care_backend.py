"""Run the shared care plan, medication diary and reminder service on localhost."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from behaviorsense.service.care_api import create_care_app
from behaviorsense.service.care_email import MailSettings
from behaviorsense.service.care_store import CareStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--database", type=Path, default=Path(os.environ.get("BS_CARE_DATABASE", ROOT / "data/care.sqlite3")))
    args = parser.parse_args()
    import uvicorn
    mail = MailSettings.from_environment()
    app = create_care_app(CareStore(args.database), ROOT / "web", mail)
    print(f"BehaviorSense Care: http://{args.host}:{args.port}")
    print(f"Email mode: {mail.mode}. Care profiles require an access code.")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
