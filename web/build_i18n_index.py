# -*- coding: utf-8 -*-
"""Generates index.html with correct UTF-8 Korean (run: python build_i18n_index.py)."""
from pathlib import Path

KO = {
    "title": "\ud55c\uad6d \ubcf5\uad8c \ubc88\ud638 \uc608\uce21\uae30 \u2014 Supabase + LSTM",
    "h1": "\ud55c\uad6d \ubcf5\uad8c \ubc88\ud638 \uc608\uce21\uae30",
    "subtitle": "TensorFlow.js LSTM + Supabase + 5\uc138\ud2b8 \ucd94\ucd9c (\ube0c\ub77c\uc6b0\uc800)",
    "notice": (
        "<strong>\uc124\uc815:</strong> "
        '<code class="text-amber-100">web/js/config.js</code>\uc5d0 Supabase '
        "<strong>anon public</strong> \ud0a4\ub97c \ub123\uc5b4 \uc8fc\uc138\uc694. "
        "<strong>service_role</strong> \ud0a4\ub294 \ube0c\ub77c\uc6b0\uc800\uc5d0 "
        "\ub123\uc73c\uba74 \uc548 \ub429\ub2c8\ub2e4. "
        "RLS\uac00 \uc788\uc73c\uba74 Supabase \ub300\uc2dc\ubcf4\ub4dc "
        "<strong>SQL Editor</strong>\uc5d0\uc11c "
        '<code class="text-amber-100">web/supabase-rls.sql</code> '
        "\ub0b4\uc6a9\uc744 \uc2e4\ud589\ud558\uc138\uc694."
    ),
    "status": "\uc0c1\ud0dc",
    "refresh": "\uc0c8\ub85c\uace0\uce68",
    "sec_input": "\ud68c\ucc28 \uc785\ub825 (\ub2f9\ucca8 \ubc88\ud638 6\uac1c)",
    "label_round": "\ud68c\ucc28",
    "label_nums": "\ubc88\ud638 6\uac1c (\uacf5\ubc31/\ucf64\ub9c8)",
    "btn_save": "\uc800\uc7a5 / \uc218\uc815",
    "sec_predict": "\ub2e4\uc74c \ud68c\ucc28 \uc608\uce21 (5\uc138\ud2b8)",
    "predict_hint": (
        "\ucd5c\ub300 \ud68c\ucc28\uac00 100\uc778 \ub3d9\uc548\uc5d0\ub294 101\ud68c\ub9cc "
        "(\uaddc\uce59\uc0c1) \uc608\uce21\ud569\ub2c8\ub2e4. "
        "\uadf8 \uc678\uc5d0\ub294 \ud56d\uc0c1 (\ucd5c\ub300 \ud68c\ucc28 + 1)\ud68c\uc785\ub2c8\ub2e4."
    ),
    "btn_lstm": "LSTM \ud559\uc2b5 \ud6c4 5\uc138\ud2b8 \uc0dd\uc131",
    "sec_result": "\uc2e4\uc81c \ub2f9\ucca8 \ubc88\ud638 \xb7 \uc77c\uce58 \ud45c\uc2dc",
    "label_actual": "\uc2e4\uc81c \ubc88\ud638 6\uac1c",
    "btn_apply": "\ube44\uad50 \ud6c4 DB \ubc18\uc601",
    "footer": (
        "\ubcf5\uad8c\uc740 \ub9e4 \ud68c\ucc28 \ub3c5\ub9bd \ucd94\ucc98\uc785\ub2c8\ub2e4. "
        "\uc608\uce21\uc740 \ucc38\uace0\uc6a9\uc774\uba70 \ub2f9\ucca8\uc744 \ubcf4\uc7a5\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4."
    ),
}


def main() -> None:
    root = Path(__file__).resolve().parent
    html = f"""<!DOCTYPE html>
<html lang="ko">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{KO["title"]}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2/dist/umd/supabase.js"></script>
  </head>
  <body class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-slate-100">
    <div class="mx-auto max-w-3xl px-4 py-10">
      <header class="mb-10 text-center">
        <h1 class="text-3xl font-bold tracking-tight text-white">{KO["h1"]}</h1>
        <p class="mt-2 text-sm text-slate-300">{KO["subtitle"]}</p>
      </header>

      <section class="mb-8 rounded-2xl border border-slate-700/80 bg-slate-900/50 p-6 shadow-xl">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <h2 class="text-lg font-semibold text-white">{KO["status"]}</h2>
          <button type="button" id="btn-refresh" class="rounded-lg bg-slate-700 px-4 py-2 text-sm font-medium hover:bg-slate-600">{KO["refresh"]}</button>
        </div>
        <div id="status" class="mt-4 text-sm text-slate-300"></div>
      </section>

      <section class="mb-8 rounded-2xl border border-slate-700/80 bg-slate-900/50 p-6 shadow-xl">
        <h2 class="text-lg font-semibold text-white">{KO["sec_input"]}</h2>
        <form id="form-input" class="mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <label class="block text-xs text-slate-400">{KO["label_round"]}</label>
            <input id="in-round" type="number" min="1" required class="mt-1 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2" />
          </div>
          <div class="sm:col-span-2">
            <label class="block text-xs text-slate-400">{KO["label_nums"]}</label>
            <input id="in-nums" type="text" placeholder="3 11 15 23 35 44" class="mt-1 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2" />
          </div>
          <div class="sm:col-span-2">
            <button type="submit" class="w-full rounded-lg bg-indigo-600 py-2.5 font-medium hover:bg-indigo-500 sm:w-auto sm:px-8">{KO["btn_save"]}</button>
          </div>
        </form>
        <p id="log" class="mt-3 text-sm text-amber-200/90"></p>
      </section>

      <section class="mb-8 rounded-2xl border border-slate-700/80 bg-slate-900/50 p-6 shadow-xl">
        <h2 class="text-lg font-semibold text-white">{KO["sec_predict"]}</h2>
        <p class="mt-1 text-xs text-slate-400">{KO["predict_hint"]}</p>
        <button type="button" id="btn-predict" class="mt-4 rounded-lg bg-emerald-700 px-6 py-2.5 font-medium hover:bg-emerald-600">{KO["btn_lstm"]}</button>
        <p id="predict-log" class="mt-3 text-sm text-slate-300"></p>
        <div id="sets-out" class="mt-4 space-y-2"></div>
      </section>

      <section class="rounded-2xl border border-slate-700/80 bg-slate-900/50 p-6 shadow-xl">
        <h2 class="text-lg font-semibold text-white">{KO["sec_result"]}</h2>
        <form id="form-result" class="mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <label class="block text-xs text-slate-400">{KO["label_round"]}</label>
            <input id="res-round" type="number" min="1" required class="mt-1 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2" />
          </div>
          <div class="sm:col-span-2">
            <label class="block text-xs text-slate-400">{KO["label_actual"]}</label>
            <input id="res-nums" type="text" placeholder="1 9 12 23 39 43" class="mt-1 w-full rounded-lg border border-slate-600 bg-slate-800 px-3 py-2" />
          </div>
          <div class="sm:col-span-2">
            <button type="submit" class="w-full rounded-lg bg-violet-600 py-2.5 font-medium hover:bg-violet-500 sm:w-auto sm:px-8">{KO["btn_apply"]}</button>
          </div>
        </form>
        <div id="result-out" class="mt-4 text-sm text-slate-200"></div>
      </section>

      <footer class="mt-10 text-center text-xs text-slate-500">{KO["footer"]}</footer>
    </div>
    <script src="js/config.js"></script>
    <script src="js/app.js"></script>
  </body>
</html>
"""
    (root / "index.html").write_text(html, encoding="utf-8")
    print("Wrote index.html (UTF-8)")

    cfg_js = """/**
 * Supabase \u2192 Settings \u2192 API \u2192 Project API keys \u2192 anon public \ud0a4\ub9cc \ub123\uc73c\uc138\uc694.
 * service_role \ud0a4\ub294 \uc808\ub300 \ub123\uc9c0 \ub9c8\uc138\uc694. (\ube0c\ub77c\uc6b0\uc800\uc5d0 \ub178\ucd9c\ub429\ub2c8\ub2e4.)
 */
window.LOTTO_CONFIG = {
  supabaseUrl: 'https://zuotnhpnanimjotspxfi.supabase.co',
  supabaseAnonKey: '',
};
"""
    (root / "js" / "config.js").write_text(cfg_js, encoding="utf-8")

    ex_js = """/**
 * config.example.js\ub97c \ubcf5\uc0ac\ud574 config.js\ub85c \uc4f0\uac70\ub098, anon public \ud0a4\ub9cc \uc785\ub825\ud558\uc138\uc694.
 * service_role\uc740 \ube0c\ub77c\uc6b0\uc800\uc5d0 \ub450\uc9c0 \ub9c8\uc138\uc694.
 */
window.LOTTO_CONFIG = {
  supabaseUrl: 'https://zuotnhpnanimjotspxfi.supabase.co',
  supabaseAnonKey: '',
};
"""
    (root / "js" / "config.example.js").write_text(ex_js, encoding="utf-8")
    print("Wrote js/config.js and js/config.example.js (UTF-8)")


if __name__ == "__main__":
    main()
