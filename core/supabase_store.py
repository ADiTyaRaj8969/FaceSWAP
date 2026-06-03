"""
Supabase-backed storage for the curated target (location) images.

Everything goes through Supabase's REST APIs with the SERVICE_ROLE key:
  - location records  -> Postgres table `locations` (via PostgREST /rest/v1)
  - target images     -> Storage bucket `location-images` (via /storage/v1)

Enabled only when SUPABASE_URL and SUPABASE_SERVICE_KEY are set; otherwise
`is_enabled()` returns False and web_app.py falls back to the local Location/
folder. Uses `requests` (already a dependency) — no extra package needed.

Run supabase_schema.sql once in the Supabase SQL editor to create the table,
bucket and read policies.
"""
import os
import re

import requests

SUPABASE_URL = (os.environ.get("SUPABASE_URL") or "").rstrip("/")
SUPABASE_KEY = (os.environ.get("SUPABASE_SERVICE_KEY")
                or os.environ.get("SUPABASE_KEY") or "")
BUCKET       = os.environ.get("SUPABASE_BUCKET", "location-images")

_TIMEOUT = 20


def is_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _rest_headers(extra: dict | None = None) -> dict:
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def _storage_headers(content_type: str | None = None) -> dict:
    h = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}
    if content_type:
        h["Content-Type"] = content_type
    return h


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "location"


def _image_path(gender: str, name: str) -> str:
    return f"{gender}/{_slug(name)}.jpg"


# ── locations table ───────────────────────────────────────────────────────────

def list_locations(gender: str) -> list:
    """
    [{folder, label, has_image, message}] for a gender, ordered by sort_order.
    Resilient to the `message` column not existing yet (older schema): falls
    back to a select without it instead of returning nothing.
    """
    def _query(select):
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/locations",
            headers=_rest_headers(),
            params={"gender": f"eq.{gender}", "order": "sort_order.asc,name.asc",
                    "select": select},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    try:
        try:
            rows = _query("name,image_path,sort_order,message")
        except Exception:
            # `message` column missing (schema not migrated) — retry without it.
            rows = _query("name,image_path,sort_order")
        return [{"folder": row["name"], "label": row["name"],
                 "has_image": bool(row.get("image_path")),
                 "message": row.get("message") or ""} for row in rows]
    except Exception as e:
        print(f"[supabase] list_locations failed: {e}")
        return []


def _get_row(gender: str, name: str):
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/locations",
        headers=_rest_headers(),
        params={"gender": f"eq.{gender}", "name": f"eq.{name}",
                "select": "id,name,image_path,sort_order", "limit": "1"},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    rows = r.json()
    return rows[0] if rows else None


def _next_sort_order(gender: str) -> int:
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/locations",
        headers=_rest_headers(),
        params={"gender": f"eq.{gender}", "order": "sort_order.desc",
                "select": "sort_order", "limit": "1"},
        timeout=_TIMEOUT,
    )
    r.raise_for_status()
    rows = r.json()
    return (rows[0]["sort_order"] + 1) if rows else 1


# ── storage ───────────────────────────────────────────────────────────────────

def _upload_image(path: str, data: bytes):
    """Upsert raw JPEG bytes to the bucket at `path`."""
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path}"
    r = requests.post(url, headers=_storage_headers("image/jpeg") | {"x-upsert": "true"},
                      data=data, timeout=_TIMEOUT)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"storage upload {r.status_code}: {r.text[:200]}")


def _delete_image(path: str):
    url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path}"
    try:
        requests.delete(url, headers=_storage_headers(), timeout=_TIMEOUT)
    except Exception as e:
        print(f"[supabase] delete image failed: {e}")


def get_image_bytes(gender: str, name: str):
    """Raw bytes of a location's target image, or None."""
    try:
        row = _get_row(gender, name)
        if not row or not row.get("image_path"):
            return None
        url = f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{row['image_path']}"
        r = requests.get(url, headers=_storage_headers(), timeout=_TIMEOUT)
        if r.status_code == 200:
            return r.content
        # public-bucket fallback
        pub = f"{SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{row['image_path']}"
        r = requests.get(pub, timeout=_TIMEOUT)
        return r.content if r.status_code == 200 else None
    except Exception as e:
        print(f"[supabase] get_image_bytes failed: {e}")
        return None


def has_image(gender: str, name: str) -> bool:
    try:
        row = _get_row(gender, name)
        return bool(row and row.get("image_path"))
    except Exception:
        return False


# ── mutations (owner / Control Panel) ─────────────────────────────────────────

def upsert_location(gender: str, name: str, image_bytes: bytes,
                    sort_order: int | None = None) -> dict:
    """
    Create or update a location and store its image. `name` is the clean label.
    If sort_order is given it's applied (preserves an explicit ordering, e.g. the
    folder number during migration); otherwise new rows get max+1. Returns
    {folder, label}.
    """
    path = _image_path(gender, name)
    _upload_image(path, image_bytes)

    row = _get_row(gender, name)
    if row:
        patch: dict[str, object] = {"image_path": path}
        if sort_order is not None:
            patch["sort_order"] = sort_order
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/locations",
            headers=_rest_headers({"Prefer": "return=minimal"}),
            params={"gender": f"eq.{gender}", "name": f"eq.{name}"},
            json=patch,
            timeout=_TIMEOUT,
        ).raise_for_status()
    else:
        requests.post(
            f"{SUPABASE_URL}/rest/v1/locations",
            headers=_rest_headers({"Prefer": "return=minimal"}),
            json={"gender": gender, "name": name, "image_path": path,
                  "sort_order": sort_order if sort_order is not None
                                else _next_sort_order(gender)},
            timeout=_TIMEOUT,
        ).raise_for_status()

    return {"folder": name, "label": name}


def rename_location(gender: str, name: str, new_name: str) -> dict:
    """Rename a location row (image stays at its current path)."""
    row = _get_row(gender, name)
    if not row:
        raise RuntimeError("Location not found.")
    if _get_row(gender, new_name):
        raise RuntimeError("A location with that name already exists.")
    requests.patch(
        f"{SUPABASE_URL}/rest/v1/locations",
        headers=_rest_headers({"Prefer": "return=minimal"}),
        params={"gender": f"eq.{gender}", "name": f"eq.{name}"},
        json={"name": new_name},
        timeout=_TIMEOUT,
    ).raise_for_status()
    return {"folder": new_name, "label": new_name}


def set_message(gender: str, name: str, message: str):
    """Set/clear the owner's custom result message for a location."""
    if _get_row(gender, name) is None:
        raise RuntimeError("Location not found.")
    requests.patch(
        f"{SUPABASE_URL}/rest/v1/locations",
        headers=_rest_headers({"Prefer": "return=minimal"}),
        params={"gender": f"eq.{gender}", "name": f"eq.{name}"},
        json={"message": (message or "").strip() or None},
        timeout=_TIMEOUT,
    ).raise_for_status()


def delete_location(gender: str, name: str):
    """Delete the row and its image."""
    row = _get_row(gender, name)
    if row and row.get("image_path"):
        _delete_image(row["image_path"])
    requests.delete(
        f"{SUPABASE_URL}/rest/v1/locations",
        headers=_rest_headers({"Prefer": "return=minimal"}),
        params={"gender": f"eq.{gender}", "name": f"eq.{name}"},
        timeout=_TIMEOUT,
    ).raise_for_status()
