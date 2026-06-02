-- ════════════════════════════════════════════════════════════════════════════
-- DeepFace Studio — Supabase schema
-- Run this once in your Supabase project:  SQL Editor → New query → paste → Run
-- ════════════════════════════════════════════════════════════════════════════

-- ── 1. Locations table ──────────────────────────────────────────────────────
-- One row per (gender, location). image_path points at the file in the
-- 'location-images' Storage bucket; NULL means "no photo uploaded yet".
create table if not exists public.locations (
  id          uuid primary key default gen_random_uuid(),
  gender      text not null check (gender in ('Male', 'Female')),
  name        text not null,
  sort_order  int  not null default 0,
  image_path  text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  unique (gender, name)
);

create index if not exists locations_gender_order_idx
  on public.locations (gender, sort_order);

-- keep updated_at fresh
create or replace function public.touch_updated_at()
returns trigger language plpgsql as $$
begin new.updated_at = now(); return new; end; $$;

drop trigger if exists locations_touch on public.locations;
create trigger locations_touch before update on public.locations
  for each row execute function public.touch_updated_at();

-- ── 2. Storage bucket for the target images (public read) ────────────────────
insert into storage.buckets (id, name, public)
values ('location-images', 'location-images', true)
on conflict (id) do nothing;

-- ── 3. Row-level security ────────────────────────────────────────────────────
-- The backend uses the SERVICE_ROLE key, which bypasses RLS, so it can
-- insert/update/delete freely. We only need public READ policies so the
-- app page (and anyone) can list locations and view images.
alter table public.locations enable row level security;

drop policy if exists "locations public read" on public.locations;
create policy "locations public read"
  on public.locations for select
  using (true);

drop policy if exists "location images public read" on storage.objects;
create policy "location images public read"
  on storage.objects for select
  using (bucket_id = 'location-images');

-- ── 4. Rows are NOT seeded here ──────────────────────────────────────────────
-- Locations are created from the photos you actually uploaded, by running:
--   python scripts/seed_supabase_locations.py
-- Only folders that contain a photo become rows (resequenced 1..N per gender),
-- and the Control Panel adds/updates more later. This avoids empty placeholder
-- locations showing up on the app page.
