-- Run in Supabase SQL Editor so the browser (anon key) can use the web app.
-- For a personal project only; tighten policies for production.

alter table lotto_history enable row level security;
alter table lotto_predictions enable row level security;

drop policy if exists "lotto_history_anon_all" on lotto_history;
create policy "lotto_history_anon_all"
  on lotto_history for all
  to anon
  using (true)
  with check (true);

drop policy if exists "lotto_predictions_anon_all" on lotto_predictions;
create policy "lotto_predictions_anon_all"
  on lotto_predictions for all
  to anon
  using (true)
  with check (true);
