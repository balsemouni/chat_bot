"""
HubSpot Client - same logic as your hubspot_integration.py,
but as a lightweight client used only by the Gateway.
Runs saves in background threads so the async event loop is never blocked.
"""
import os, time, json, io, requests, threading
from datetime import datetime
from typing import Optional


class HubSpotClient:
    DEFAULT_EMAIL = os.getenv("HUBSPOT_CONTACT_EMAIL", "ounibalsem@gmail.com")
    DEFAULT_FIRST = os.getenv("HUBSPOT_CONTACT_FIRST", "Oun")
    DEFAULT_LAST  = os.getenv("HUBSPOT_CONTACT_LAST", "Balsem")

    def __init__(self):
        self.token = os.getenv("HUBSPOT_API_KEY")
        self.base_url = "https://api.hubapi.com"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self._sessions: dict = {}
        self.contact_id: Optional[str] = None

        if self.token:
            threading.Thread(target=self._init_contact, daemon=True).start()
        else:
            print("⚠️  HUBSPOT_API_KEY not set - logging disabled")

    def _init_contact(self):
        try:
            cid = self._find_contact(self.DEFAULT_EMAIL)
            if not cid:
                cid = self._create_contact(
                    self.DEFAULT_EMAIL, self.DEFAULT_FIRST, self.DEFAULT_LAST)
            self.contact_id = cid
            print(f"✅ HubSpot: {self.DEFAULT_EMAIL} ({cid})")
        except Exception as e:
            print(f"❌ HubSpot init: {e}")

    def start_session(self, session_id: str):
        self._sessions[session_id] = {
            "start_ts": time.time(), "utterances": []}

    def add_utterance(self, session_id: str, speaker_id: str,
                      speaker_name: str, text: str):
        if session_id not in self._sessions:
            self.start_session(session_id)
        s = self._sessions[session_id]
        elapsed_ms = int((time.time() - s["start_ts"]) * 1000)
        s["utterances"].append({
            "speaker": {"id": speaker_id, "name": speaker_name},
            "text": text,
            "startTimeMillis": elapsed_ms,
            "endTimeMillis": elapsed_ms + 1000
        })

    def end_session(self, session_id: str, transcript: str):
        if not self.token or not self.contact_id:
            return
        s = self._sessions.pop(session_id, {})
        if not s:
            return
        threading.Thread(
            target=self._save_to_hubspot, args=(s, transcript),
            daemon=True).start()

    # ---- same API helpers as hubspot_integration.py ----
    def _save_to_hubspot(self, session_data: dict, summary: str):
        try:
            file_id = self._upload_file(summary)
            call_id = self._create_call(summary, session_data["start_ts"], file_id)
            if call_id:
                self._upload_transcript(call_id, session_data["utterances"])
                print(f"✅ HubSpot call saved: {call_id}")
        except Exception as e:
            print(f"❌ HubSpot save failed: {e}")

    def _find_contact(self, email):
        r = requests.post(
            f"{self.base_url}/crm/v3/objects/contacts/search",
            headers=self.headers,
            json={"filterGroups": [{"filters": [
                {"propertyName": "email", "operator": "EQ", "value": email}
            ]}]})
        results = r.json().get("results", [])
        return results[0]["id"] if results else None

    def _create_contact(self, email, first, last):
        r = requests.post(
            f"{self.base_url}/crm/v3/objects/contacts",
            headers=self.headers,
            json={"properties": {"email": email,
                                  "firstname": first, "lastname": last}})
        return r.json().get("id") if r.status_code == 201 else None

    def _upload_file(self, text: str):
        fn = f"Voice_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        r = requests.post(
            "https://api.hubapi.com/files/v3/files",
            headers={"Authorization": f"Bearer {self.token}"},
            files={
                "file": (fn, text.encode(), "text/plain"),
                "options": (None, json.dumps({
                    "access": "PRIVATE", "overwrite": "false",
                    "duplicateValidationStrategy": "NONE",
                    "duplicateValidationScope": "ENTIRE_PORTAL"
                }), "application/json")
            })
        return r.json().get("id") if r.status_code == 201 else None

    def _create_call(self, summary: str, start_ts: float, file_id=None):
        props = {
            "hs_call_title": f"Voice Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "hs_call_direction": "INBOUND",
            "hs_call_disposition": "f240bbac-87c9-4f6e-bf70-924b57d47db7",
            "hs_timestamp": datetime.utcnow().isoformat() + "Z",
            "hs_call_body": summary[:5000],
            "hs_call_app_id": "30746066",
            "hs_call_duration": max(1, int((time.time() - start_ts) / 60))
        }
        if file_id:
            props["hs_attachment_ids"] = str(file_id)
        r = requests.post(
            f"{self.base_url}/crm/v3/objects/calls",
            headers=self.headers,
            json={"properties": props, "associations": [{
                "to": {"id": self.contact_id},
                "types": [{"associationCategory": "HUBSPOT_DEFINED",
                           "associationTypeId": 194}]
            }]})
        return r.json().get("id") if r.status_code == 201 else None

    def _upload_transcript(self, call_id: str, utterances: list):
        requests.post(
            f"{self.base_url}/crm/v3/extensions/calling/transcripts",
            headers=self.headers,
            json={"engagementId": call_id,
                  "transcriptCreateUtterances": utterances})