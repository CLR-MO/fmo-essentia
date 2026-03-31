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
        batch_size: int = 1000,
    ):
        """Return (total, generator) for all audio entities, paged by batch_size.

        Fetches page 1 immediately so total is known before iteration begins.
        Always filters by '@filetype:=:audio' (using the entity column, so
        .webm audio is included correctly).  When a query is given it is
        AND'd with the filetype filter so non-audio entities (images, docs)
        are never returned even if the user query would match them.
        """
        effective_query = f"@filetype:=:audio & ({query})" if query else "@filetype:=:audio"

        first = self.query_entities(
            query=effective_query,
            fields="id,path,name,filetype",
            page=1,
            limit=batch_size,
            db=db,
        )
        pagination = first.get("pagination", {})
        total = pagination.get("total", 0)
        total_pages = pagination.get("totalPages", 1)

        def _gen():
            yield from first.get("entities", [])
            for page in range(2, total_pages + 1):
                result = self.query_entities(
                    query=effective_query,
                    fields="id,path,name,filetype",
                    page=page,
                    limit=batch_size,
                    db=db,
                )
                yield from result.get("entities", [])

        return total, _gen()

    # ── Read single entity ──────────────────────────────────────────────────────

    def get_entity(self, id: int, db: str | None = None) -> dict:
        """GET /entities/:id"""
        params: dict = {}
        if db:
            params["db"] = db
        return self._get(f"/entities/{id}", **params)

    # ── Write ─────────────────────────────────────────────────────────────────

    def bulk_update(
        self,
        updates: list[dict],
        db: str | None = None,
        add_only: bool = False,
    ) -> dict:
        """
        POST /entities/bulk/update or /entities/bulk/add-attributes

        updates: [{"id": 123, "tags": ["/ mood / happy"], "attributes": {"bpm": 120}}, ...]
        add_only: if True, only add new attributes without overwriting existing ones
        """
        body: dict = {"updates": updates}
        if db:
            body["db"] = db
        path = "/entities/bulk/add-attributes" if add_only else "/entities/bulk/update"
        return self._post(path, body)
