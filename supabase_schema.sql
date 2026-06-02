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

-- ── 4. (Optional) seed the 19 standard locations for both genders ────────────
-- Photos are added later via the Control Panel; these just pre-create the rows.
insert into public.locations (gender, name, sort_order)
select g.gender, l.name, l.ord
from   (values ('Male'), ('Female')) as g(gender)
cross join (values
  (1,'Marwadi University'),(2,'ICT Department'),(3,'Library'),
  (4,'Electronic Circuit Lab'),(5,'Embedded System Lab'),
  (6,'Data Science And AI Lab'),(7,'Ideation Lab'),(8,'IoT Lab'),
  (9,'VLSI Lab'),(10,'Web Development Lab'),(11,'Project Lab'),
  (12,'Programming Lab'),(13,'MUIIR'),(14,'Classroom'),
  (15,'Sports Ground'),(16,'Field'),(17,'Music Room'),
  (18,'Library G Floor'),(19,'Ground')
) as l(ord, name)
on conflict (gender, name) do nothing;
