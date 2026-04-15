-- ─────────────────────────────────────────────────────────────────────────────
-- Nigerian Real Estate — Supabase Schema
-- Run this in the Supabase SQL Editor (project → SQL Editor → New query)
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable the pgcrypto extension for gen_random_uuid() (already on in most projects)
create extension if not exists "pgcrypto";

-- ── predictions table ─────────────────────────────────────────────────────────
create table if not exists public.predictions (
    id                   uuid primary key default gen_random_uuid(),
    created_at           timestamptz      not null default now(),

    -- Optional: tie to a Supabase Auth user. Remove or set nullable if no auth.
    user_id              uuid             references auth.users (id) on delete set null,

    -- Top-level identifiers (duplicated from inputs for easy querying)
    city                 text             not null,
    neighborhood         text             not null,

    -- Full request payload stored as JSON for auditability / future retraining
    inputs               jsonb            not null,

    -- Regression output
    annual_rent_ngn      double precision not null,
    monthly_rent_ngn     double precision not null,

    -- Property type classification output
    property_type        text             not null,
    probabilities_type   jsonb            not null,

    -- Price tier classification output
    price_tier           text             not null,   -- Low | Mid | High
    probabilities_tier   jsonb            not null
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
create index if not exists predictions_user_id_idx    on public.predictions (user_id);
create index if not exists predictions_created_at_idx on public.predictions (created_at desc);
create index if not exists predictions_city_idx       on public.predictions (city);

-- ── Row Level Security ────────────────────────────────────────────────────────
alter table public.predictions enable row level security;

-- Policy: authenticated users can insert their own predictions
create policy "Users can insert own predictions"
    on public.predictions
    for insert
    to authenticated
    with check (auth.uid() = user_id);

-- Policy: authenticated users can read their own predictions
create policy "Users can read own predictions"
    on public.predictions
    for select
    to authenticated
    using (auth.uid() = user_id);

-- ── Optional: allow anonymous inserts (remove if you require auth) ────────────
-- Uncomment the block below if you want unauthenticated users to save predictions
-- (set user_id to null in your server action when no session exists)

-- create policy "Anon can insert with null user_id"
--     on public.predictions
--     for insert
--     to anon
--     with check (user_id is null);
