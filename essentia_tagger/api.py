"""Thin client for the CLRMO REST API."""

import json
from typing import Any

import requests


class ClrmoClient:
    def __init__(self, base_url: str = "http://127.0.0.1:3283", timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    def _get(self, path: str, **params) -> Any:
        r = self._session.get(f"{self.base}{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: Any) -> Any:
        r = self._session.post(f"{self.base}{path}", data=json.dumps(body), timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ── Read ──────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return self._get("/status")

    def query_entities(
        self,
        query: str = "",
        fields: str = "id,path,name,filetype",
        page: int = 1,
        limit: int = 500,
        db: str | None = None,
    ) -> dict:
        params: dict = {"fields": fields, "page": page, "limit": limit}
        if query:
            params["query"] = query
        if db:
            params["db"] = db
        return self._get("/entities", **params)

    def iter_audio_entities(
        self,
        query: str = "",
        db: str | None = None,
        batch_size: int = 500,
    ):
        """Yield all audio entities page by page.

        Always filters by '@filetype:=:audio' (using the entity column, so
        .webm audio is included correctly).  When a query is given it is
        AND'd with the filetype filter so non-audio entities (images, docs)
        are never returned even if the user query would match them.
        """
        effective_query = f"@filetype:=:audio & ({query})" if query else "@filetype:=:audio"
        page = 1
        while True:
            result = self.query_entities(
                query=effective_query,
                fields="id,path,name,filetype",
                page=page,
                limit=batch_size,
                db=db,
            )
            entities = result.get("entities", [])
            if not entities:
                break
            yield from entities
            total_pages = result.get("pagination", {}).get("totalPages", 1)
            if page >= total_pages:
                break
            page += 1

    # ── Write ─────────────────────────────────────────────────────────────────

    def bulk_update(
        self,
        updates: list[dict],
        db: str | None = None,
    ) -> dict:
        """
        POST /entities/bulk/update

        updates: [{"id": 123, "tags": ["/ mood / happy"], "attributes": {"bpm": 120}}, ...]
        """
        body: dict = {"updates": updates}
        if db:
            body["db"] = db
        return self._post("/entities/bulk/update", body)
