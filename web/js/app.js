(function () {
  'use strict';

  const MIN_FOR_FIRST_PREDICT = 5;

  // ── Supabase client ──────────────────────────────────────────────────────
  function getClient() {
    const c = window.LOTTO_CONFIG;
    if (!c || !c.supabaseUrl || !c.supabaseAnonKey)
      throw new Error('Set supabaseUrl and supabaseAnonKey in web/js/config.js');
    const lib = window.supabase;
    if (!lib || typeof lib.createClient !== 'function')
      throw new Error('Supabase library failed to load');
    return lib.createClient(c.supabaseUrl, c.supabaseAnonKey);
  }

  // ── Seeded RNG ───────────────────────────────────────────────────────────
  function mulberry32(a) {
    return function () {
      let t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Feature Engineering
  // Per-round feature vector (135 dims, MinMax scaled):
  //   [  0.. 44] multihot  : 0/1 for each number 1-45
  //   [ 45.. 89] MA10      : rolling 10-round appearance frequency (0..1)
  //   [ 90..134] gap_norm  : rounds since last seen / 45  (0 = appeared this round)
  // ─────────────────────────────────────────────────────────────────────────
  function buildEnrichedFeatures(rounds) {
    const T   = rounds.length;
    const NF  = 135;
    const features = [];
    const cnt10    = new Float64Array(45);
    const lastSeen = new Int32Array(45).fill(-1);

    for (let t = 0; t < T; t++) {
      // update MA10 window (add current, remove 10-rounds-ago)
      for (const n of rounds[t]) cnt10[n - 1]++;
      if (t >= 10) for (const n of rounds[t - 10]) cnt10[n - 1]--;
      const w10 = Math.min(10, t + 1);

      // update lastSeen
      for (const n of rounds[t]) lastSeen[n - 1] = t;

      const feat = new Float32Array(NF);
      // 1. multihot
      for (const n of rounds[t]) feat[n - 1] = 1;
      // 2. MA10
      for (let i = 0; i < 45; i++) feat[45 + i] = cnt10[i] / w10;
      // 3. gap_norm (0 = appeared this round)
      for (let i = 0; i < 45; i++) {
        const gap = lastSeen[i] < 0 ? Math.min(t + 1, 45) : (t - lastSeen[i]);
        feat[90 + i] = Math.min(1, gap / 45);
      }
      features.push(feat);
    }

    // MinMax scale each feature column
    const mins = new Float32Array(NF).fill(Infinity);
    const maxs = new Float32Array(NF).fill(-Infinity);
    for (const f of features) {
      for (let j = 0; j < NF; j++) {
        if (f[j] < mins[j]) mins[j] = f[j];
        if (f[j] > maxs[j]) maxs[j] = f[j];
      }
    }
    for (const f of features) {
      for (let j = 0; j < NF; j++) {
        const r = maxs[j] - mins[j];
        f[j] = r > 1e-9 ? (f[j] - mins[j]) / r : 0;
      }
    }
    return { features, numFeatures: NF };
  }

  // Plain multihot (kept for patternMatchBoost heuristic)
  function roundsToMultihot(rounds) {
    return rounds.map(nums => {
      const row = new Float32Array(45);
      for (const n of nums) row[n - 1] = 1;
      return row;
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Tensor utilities
  // ─────────────────────────────────────────────────────────────────────────
  function buildTensors3d(features, seqLen) {
    const NF = features[0].length;
    const n  = features.length - seqLen;
    if (n <= 0) throw new Error('Not enough data for sequences (seqLen=' + seqLen + ')');
    const xArr = new Float32Array(n * seqLen * NF);
    const yArr = new Float32Array(n * 45);
    for (let i = 0; i < n; i++) {
      const t = i + seqLen;
      for (let s = 0; s < seqLen; s++) {
        const f = features[t - seqLen + s];
        for (let d = 0; d < NF; d++) xArr[i * seqLen * NF + s * NF + d] = f[d];
      }
      // target: multihot is the first 45 dims of the enriched feature
      for (let d = 0; d < 45; d++) yArr[i * 45 + d] = features[t][d];
    }
    return {
      xsT: tf.tensor3d(xArr, [n, seqLen, NF]),
      ysT: tf.tensor2d(yArr, [n, 45]),
      n, NF
    };
  }

  function buildLastInput3d(features, seqLen) {
    const NF  = features[0].length;
    const arr = new Float32Array(seqLen * NF);
    for (let s = 0; s < seqLen; s++) {
      const f = features[features.length - seqLen + s];
      for (let d = 0; d < NF; d++) arr[s * NF + d] = f[d];
    }
    return tf.tensor3d(arr, [1, seqLen, NF]);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Shared fit options: EarlyStopping(patience=7), LR=0.0005, shuffle=false
  // ─────────────────────────────────────────────────────────────────────────
  function makeFitOpts(modelRef, n, maxEpochs) {
    const patience = 7;
    let best = Infinity, wait = 0;
    return {
      epochs:          maxEpochs,
      batchSize:       Math.max(8, Math.min(32, Math.floor(n * 0.12))),
      validationSplit: n > 40 ? 0.15 : 0,
      shuffle:         false,   // time-series: DO NOT shuffle
      verbose:         0,
      callbacks: {
        onEpochEnd: (_ep, logs) => {
          const v = (logs.val_loss != null) ? logs.val_loss : logs.loss;
          if (v < best - 1e-4) { best = v; wait = 0; }
          else if (++wait >= patience) { modelRef.stopTraining = true; }
        }
      }
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Model 1: 2-layer LSTM  (64 → Dropout 0.2 → 32 → Dense 45)
  // ─────────────────────────────────────────────────────────────────────────
  async function trainAndPredictLstm(features, seqLen, maxEpochs) {
    const { xsT, ysT, n, NF } = buildTensors3d(features, seqLen);
    const model = tf.sequential();
    model.add(tf.layers.lstm({
      units: 64, inputShape: [seqLen, NF],
      returnSequences: true, recurrentDropout: 0.1
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.lstm({ units: 32, returnSequences: false }));
    model.add(tf.layers.dense({ units: 45, activation: 'sigmoid' }));
    model.compile({ optimizer: tf.train.adam(0.0005), loss: 'binaryCrossentropy' });

    await model.fit(xsT, ysT, makeFitOpts(model, n, maxEpochs));

    const lastT = buildLastInput3d(features, seqLen);
    const pred  = model.predict(lastT);
    const probs = Array.from(await pred.data());
    pred.dispose(); lastT.dispose(); xsT.dispose(); ysT.dispose(); model.dispose();
    return probs;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Model 2: 2-layer GRU  (48 → Dropout 0.2 → 24 → Dense 45)
  // ─────────────────────────────────────────────────────────────────────────
  async function trainAndPredictGru(features, seqLen, maxEpochs) {
    const { xsT, ysT, n, NF } = buildTensors3d(features, seqLen);
    const model = tf.sequential();
    model.add(tf.layers.gru({
      units: 48, inputShape: [seqLen, NF],
      returnSequences: true, recurrentDropout: 0.1
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.gru({ units: 24, returnSequences: false }));
    model.add(tf.layers.dense({ units: 45, activation: 'sigmoid' }));
    model.compile({ optimizer: tf.train.adam(0.0005), loss: 'binaryCrossentropy' });

    await model.fit(xsT, ysT, makeFitOpts(model, n, maxEpochs));

    const lastT = buildLastInput3d(features, seqLen);
    const pred  = model.predict(lastT);
    const probs = Array.from(await pred.data());
    pred.dispose(); lastT.dispose(); xsT.dispose(); ysT.dispose(); model.dispose();
    return probs;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Model 3: MLP / Dense  (128→Dropout→64→Dense45)
  // Uses a fixed seqMlp=10 window (flattened) to keep computation light
  // ─────────────────────────────────────────────────────────────────────────
  async function trainAndPredictMlp(features, maxEpochs) {
    const NF      = features[0].length;
    const seqMlp  = Math.min(10, features.length - 2);
    if (seqMlp < 1) return new Array(45).fill(1 / 45);
    const flatDim = seqMlp * NF;
    const nSamples = features.length - seqMlp;
    if (nSamples < 4) return new Array(45).fill(1 / 45);

    const xArr = new Float32Array(nSamples * flatDim);
    const yArr = new Float32Array(nSamples * 45);
    for (let i = 0; i < nSamples; i++) {
      for (let s = 0; s < seqMlp; s++) {
        const f = features[i + s];
        for (let d = 0; d < NF; d++) xArr[i * flatDim + s * NF + d] = f[d];
      }
      for (let d = 0; d < 45; d++) yArr[i * 45 + d] = features[i + seqMlp][d];
    }
    const xsT = tf.tensor2d(xArr, [nSamples, flatDim]);
    const ysT = tf.tensor2d(yArr, [nSamples, 45]);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [flatDim] }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 45, activation: 'sigmoid' }));
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'binaryCrossentropy' });

    const patience = 7;
    let best = Infinity, wait = 0;
    await model.fit(xsT, ysT, {
      epochs:          maxEpochs,
      batchSize:       Math.max(8, Math.min(32, Math.floor(nSamples * 0.12))),
      validationSplit: nSamples > 40 ? 0.15 : 0,
      shuffle:         false,
      verbose:         0,
      callbacks: {
        onEpochEnd: (_ep, logs) => {
          const v = (logs.val_loss != null) ? logs.val_loss : logs.loss;
          if (v < best - 1e-4) { best = v; wait = 0; }
          else if (++wait >= patience) { model.stopTraining = true; }
        }
      }
    });

    const lastFlat = new Float32Array(flatDim);
    for (let s = 0; s < seqMlp; s++) {
      const f = features[features.length - seqMlp + s];
      for (let d = 0; d < NF; d++) lastFlat[s * NF + d] = f[d];
    }
    const lastT = tf.tensor2d(lastFlat, [1, flatDim]);
    const pred  = model.predict(lastT);
    const probs = Array.from(await pred.data());
    pred.dispose(); lastT.dispose(); xsT.dispose(); ysT.dispose(); model.dispose();
    return probs;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Heuristics
  // ─────────────────────────────────────────────────────────────────────────

  // Cold/Hot weights based on recent windowN rounds
  function coldHotWeights(historyRounds, windowN) {
    const counts = new Float64Array(45);
    const recent = historyRounds.length >= windowN
      ? historyRounds.slice(-windowN) : historyRounds;
    for (const nums of recent) for (const n of nums) counts[n - 1] += 1;
    let mean = 0;
    for (let i = 0; i < 45; i++) mean += counts[i];
    mean = mean / 45 + 1e-6;
    const w = new Float64Array(45);
    for (let i = 0; i < 45; i++) {
      const v = Math.exp((-0.35 * (counts[i] - mean)) / (mean + 1e-6));
      w[i] = Math.min(3, Math.max(0.25, v));
    }
    return w;
  }

  // Cosine-similarity pattern matching boost
  function patternMatchBoost(historyRounds, seqLen) {
    if (historyRounds.length <= seqLen + 1) return new Float64Array(45).fill(1);
    const mh   = roundsToMultihot(historyRounds);
    const rows = mh.map(r => Array.from(r));
    function meanWindow(start, len) {
      const acc = new Float64Array(45);
      for (let i = 0; i < len; i++) for (let j = 0; j < 45; j++) acc[j] += rows[start + i][j];
      for (let j = 0; j < 45; j++) acc[j] /= len;
      return acc;
    }
    const target = meanWindow(rows.length - seqLen, seqLen);
    let tnorm = 0;
    for (let j = 0; j < 45; j++) tnorm += target[j] * target[j];
    tnorm = Math.sqrt(tnorm) + 1e-9;
    const boost = new Float64Array(45);
    for (let t = seqLen; t < rows.length - 1; t++) {
      const win = meanWindow(t - seqLen, seqLen);
      let wnorm = 0, dot = 0;
      for (let j = 0; j < 45; j++) { wnorm += win[j] * win[j]; dot += win[j] * target[j]; }
      const sim = dot / (Math.sqrt(wnorm) + 1e-9) / tnorm;
      if (sim > 0.92) {
        const nxt = rows[t + 1];
        for (let j = 0; j < 45; j++) boost[j] += nxt[j] * (sim - 0.92) * 10;
      }
    }
    let mx = 0;
    for (let j = 0; j < 45; j++) mx = Math.max(mx, boost[j]);
    const out = new Float64Array(45);
    if (mx > 0) for (let j = 0; j < 45; j++) out[j] = 1 + 0.6 * (boost[j] / (mx + 1e-9));
    else out.fill(1);
    return out;
  }

  // Gap weights: numbers not seen for longer get higher weight
  function gapWeights(historyRounds) {
    const size    = 45;
    const lastSeen = new Array(size).fill(historyRounds.length);
    for (let i = 0; i < size; i++) {
      for (let r = historyRounds.length - 1; r >= 0; r--) {
        if (historyRounds[r].includes(i + 1)) { lastSeen[i] = historyRounds.length - 1 - r; break; }
      }
    }
    const maxGap = Math.max(...lastSeen) + 1;
    const w = new Float64Array(size);
    for (let i = 0; i < size; i++) w[i] = 0.4 + 0.9 * (lastSeen[i] / maxGap);
    return w;
  }

  // Regression trend: compare recent vs older window frequency
  function regressionTrendWeights(historyRounds, window_, size) {
    window_ = window_ || 20;
    size    = size    || 45;
    const w = new Float64Array(size).fill(1);
    if (historyRounds.length < window_ * 2) return w;
    const recent = historyRounds.slice(-window_);
    const old    = historyRounds.slice(-window_ * 2, -window_);
    const rCnt = new Float64Array(size);
    const oCnt = new Float64Array(size);
    for (const nums of recent) for (const n of nums) rCnt[n - 1]++;
    for (const nums of old)    for (const n of nums) oCnt[n - 1]++;
    for (let i = 0; i < size; i++) {
      const trend = (rCnt[i] + 0.5) / (oCnt[i] + 0.5);
      w[i] = Math.min(2.0, Math.max(0.3, trend));
    }
    return w;
  }

  // GBM-style frequency model (XGBoost heuristic)
  function gbmScores(historyRounds, size) {
    size = size || 45;
    const n        = historyRounds.length;
    const scores   = new Float64Array(size);
    const window5  = historyRounds.slice(-5);
    const window10 = historyRounds.slice(-10);
    const window30 = historyRounds.slice(-30);
    function cnt(slice, idx) {
      let c = 0;
      for (const r of slice) if (r.includes(idx + 1)) c++;
      return c;
    }
    for (let i = 0; i < size; i++) {
      const f5  = cnt(window5,       i) / (window5.length  + 1e-9);
      const f10 = cnt(window10,      i) / (window10.length + 1e-9);
      const f30 = cnt(window30,      i) / (window30.length + 1e-9);
      const fall = cnt(historyRounds, i) / (n              + 1e-9);
      scores[i] = 0.35 * f5 + 0.30 * f10 + 0.20 * f30 + 0.15 * fall + 1e-6;
    }
    return scores;
  }

  function movingAverageSumBias(historyRounds, lookback) {
    const slice = historyRounds.slice(-lookback);
    const sums  = slice.map(r => r.reduce((a, b) => a + b, 0));
    if (sums.length < 3) return [100, 175];
    const mu   = sums.reduce((a, b) => a + b, 0) / sums.length;
    const var_ = sums.reduce((a, b) => a + (b - mu) * (b - mu), 0) / sums.length;
    const sigma = Math.sqrt(var_) + 1e-6;
    return [
      Math.max(75,  Math.floor(mu - 1.2 * sigma)),
      Math.min(220, Math.floor(mu + 1.2 * sigma))
    ];
  }

  function oddEvenOk(nums) {
    const odd = nums.filter(n => n % 2 === 1).length;
    return (odd === 2 || odd === 3 || odd === 4);
  }

  function hasExcessiveConsecutive(nums) {
    const sorted = nums.slice().sort((a, b) => a - b);
    let run = 1;
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i] === sorted[i - 1] + 1) { if (++run > 2) return true; }
      else run = 1;
    }
    return false;
  }

  function passesFilters(nums, sumLow, sumHigh, strictOE) {
    const s = nums.reduce((a, b) => a + b, 0);
    if (s < sumLow || s > sumHigh) return false;
    if (strictOE && !oddEvenOk(nums)) return false;
    if (hasExcessiveConsecutive(nums)) return false;
    return true;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Set generation helpers
  // ─────────────────────────────────────────────────────────────────────────
  function weightedPickSix(prob45, rng) {
    const picked = [];
    const p      = prob45.slice();
    for (let k = 0; k < 6; k++) {
      let s = 0;
      const avail = [];
      for (let i = 0; i < 45; i++) {
        if (!picked.includes(i)) { s += p[i]; avail.push(i); }
      }
      if (s <= 0 || avail.length === 0) break;
      let r = rng() * s, chosen = avail[0];
      for (const i of avail) { r -= p[i]; if (r <= 0) { chosen = i; break; } }
      picked.push(chosen);
    }
    return picked;
  }

  function sampleSixFromScores(scores, rng, sumLow, sumHigh) {
    const p    = scores.map(x => Math.max(x, 1e-12));
    const sumP = p.reduce((a, b) => a + b, 0);
    for (let i = 0; i < 45; i++) p[i] /= sumP;
    function draw(strictOE) {
      for (let tries = 0; tries < 8000; tries++) {
        const picked = weightedPickSix(p, rng);
        if (picked.length < 6) continue;
        const nums = picked.map(i => i + 1).sort((a, b) => a - b);
        if (passesFilters(nums, sumLow, sumHigh, strictOE)) return nums;
      }
      return null;
    }
    let r = draw(true);
    if (r) return r;
    r = draw(false);
    if (r) return r;
    const idx = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 12).map(x => x[1]);
    for (let tries = 0; tries < 8000; tries++) {
      const sh   = idx.slice().sort(() => rng() - 0.5);
      const pick = sh.slice(0, 6).sort((a, b) => a - b);
      const nums = pick.map(i => i + 1);
      if (passesFilters(nums, sumLow, sumHigh, false)) return nums;
    }
    return null;
  }

  function generateSets(baseScores, numSets, seed, sumLow, sumHigh) {
    const rng  = mulberry32(seed >>> 0);
    const sets = [];
    const used = new Set();
    let attempt = 0;
    while (sets.length < numSets && attempt < numSets * 400) {
      attempt++;
      const scores = baseScores.map(b => Math.max(b * Math.exp((rng() - 0.5) * 0.14), 1e-12));
      const got    = sampleSixFromScores(scores, rng, sumLow, sumHigh);
      if (!got) continue;
      const key = got.join(',');
      if (used.has(key)) continue;
      used.add(key);
      sets.push(got);
    }
    while (sets.length < numSets) {
      const pool = [];
      while (pool.length < 6) { const n = 1 + Math.floor(rng() * 45); if (!pool.includes(n)) pool.push(n); }
      pool.sort((a, b) => a - b);
      const key = pool.join(',');
      if (!used.has(key)) { used.add(key); sets.push(pool); }
    }
    return sets;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // DB helpers
  // ─────────────────────────────────────────────────────────────────────────
  async function fetchHistory(client) {
    const { data, error } = await client
      .from('lotto_history').select('round_no, numbers').order('round_no', { ascending: true });
    if (error) throw error;
    return (data || []).map(r => ({
      round:   r.round_no,
      numbers: (r.numbers || []).map(Number).sort((a, b) => a - b)
    }));
  }

  function computeMatches(predictedSets, actual) {
    const aset = new Set(actual);
    return predictedSets.map((s, i) => {
      const matched = s.filter(n => aset.has(n)).sort((a, b) => a - b);
      return { set_index: i, predicted: s, matched_numbers: matched, match_count: matched.length };
    });
  }

  async function refreshStatus(client, el) {
    const rows    = await fetchHistory(client);
    const maxR    = rows.length ? Math.max(...rows.map(r => r.round)) : null;
    const next    = maxR == null ? 1 : maxR + 1;
    const maxLabel = maxR == null ? '\u2014' : String(maxR);
    el.innerHTML =
      '<p>\uc800\uc7a5\ub41c \ud68c\ucc28: <strong>' + rows.length +
      '</strong>\uac1c</p><p>\ucd5c\ub300 \ud68c\ucc28: <strong>' + maxLabel +
      '</strong> \u2192 \ub2e4\uc74c \uc608\uce21 \ub300\uc0c1: <strong>' + next + '\ud68c</strong></p>';
    const resRoundInput = document.getElementById('res-round');
    if (resRoundInput) resRoundInput.value = String(next);
    return { rows, maxR, next };
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Main
  // ─────────────────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', async () => {
    const statusEl  = document.getElementById('status');
    const logEl     = document.getElementById('log');
    const predictLog = document.getElementById('predict-log');
    const setsOut   = document.getElementById('sets-out');
    let client;

    try { client = getClient(); }
    catch (e) {
      statusEl.innerHTML = '<p class="text-red-400">' + (e.message || String(e)) + '</p>';
      return;
    }

    document.getElementById('btn-refresh').onclick = async () => {
      try { await refreshStatus(client, statusEl); logEl.textContent = ''; }
      catch (e) { logEl.textContent = e.message || String(e); }
    };

    await refreshStatus(client, statusEl);

    // ── 번호 입력 ──────────────────────────────────────────────────────────
    document.getElementById('form-input').onsubmit = async (ev) => {
      ev.preventDefault();
      const round = Number(document.getElementById('in-round').value);
      const raw   = document.getElementById('in-nums').value.trim().split(/[\s,]+/);
      const nums  = raw.map(Number).filter(n => !Number.isNaN(n));
      nums.sort((a, b) => a - b);
      if (nums.length !== 6 || new Set(nums).size !== 6) { logEl.textContent = 'Enter exactly 6 distinct numbers.'; return; }
      if (nums.some(n => n < 1 || n > 45))               { logEl.textContent = 'Numbers must be 1\u201345.'; return; }
      try {
        const { error } = await client.from('lotto_history')
          .upsert({ round_no: round, numbers: nums }, { onConflict: 'round_no' });
        if (error) throw error;
        logEl.textContent = 'Saved round ' + round + ': ' + nums.join(', ');
        await refreshStatus(client, statusEl);
      } catch (e) {
        logEl.textContent = (e.message || String(e)) + ' \u2014 check anon key and RLS on lotto_history.';
      }
    };

    // ── 예측 버튼 ──────────────────────────────────────────────────────────
    document.getElementById('btn-predict').onclick = async () => {
      setsOut.innerHTML    = '';
      predictLog.textContent = '';
      try {
        const { rows, maxR, next } = await refreshStatus(client, statusEl);
        if (!rows.length) {
          predictLog.textContent = '\ud68c\ucc28 \ub370\uc774\ud130\ub97c \uba3c\uc800 \uc785\ub825\ud558\uc138\uc694.';
          return;
        }
        if (rows.length < MIN_FOR_FIRST_PREDICT) {
          predictLog.textContent =
            '\uc608\uce21\ud558\ub824\uba74 \ucd5c\uc18c ' + MIN_FOR_FIRST_PREDICT +
            '\ud68c\ucc28 \uc774\uc0c1 \uc785\ub825\uc774 \ud544\uc694\ud569\ub2c8\ub2e4. (\ud604\uc7ac ' + rows.length + '\ud68c\ucc28)';
          return;
        }
        const numsOnly = rows.map(r => r.numbers);
        if (numsOnly.length < 2) {
          predictLog.textContent = '\uc608\uce21\ud558\ub824\uba74 \ucd5c\uc18c 2\ud68c\ucc28 \uc774\uc0c1 \ud544\uc694\ud569\ub2c8\ub2e4.';
          return;
        }

        const n = numsOnly.length;

        // ── 회차 수에 따른 동적 seqLen·maxEpoch 설정 ─────────────────────
        // 데이터 많을수록 seqLen 길게 (패턴 더 잘 잡음), epoch는 EarlyStopping으로 자동 조기 종료
        let dynamicSeqLen, dynamicMaxEpochs;
        if      (n <= 30)  { dynamicSeqLen = 8;  dynamicMaxEpochs = 100; }
        else if (n <= 100) { dynamicSeqLen = 12; dynamicMaxEpochs = 80;  }
        else if (n <= 200) { dynamicSeqLen = 20; dynamicMaxEpochs = 60;  }
        else if (n <= 350) { dynamicSeqLen = 24; dynamicMaxEpochs = 50;  }
        else               { dynamicSeqLen = 30; dynamicMaxEpochs = 40;  }
        dynamicSeqLen = Math.min(dynamicSeqLen, n - 1);

        // ── 피처 엔지니어링 (MinMaxScaler + MA10 + Gap) ───────────────────
        predictLog.textContent =
          '\ud53c\ucc98 \uc5d4\uc9c0\ub2c8\uc5b4\ub9c1 \uc911\u2026 (MinMax \uc815\uaddc\ud654 + MA10 + Gap | 135\ucc28\uc6d0)';
        const { features } = buildEnrichedFeatures(numsOnly);

        // ── Model 1: LSTM ─────────────────────────────────────────────────
        predictLog.textContent =
          '[1/3] LSTM \ud559\uc2b5 \uc911\u2026 (' + n + '\ud68c\ucc28, seq=' +
          dynamicSeqLen + ', max ' + dynamicMaxEpochs + 'ep, EarlyStopping)';
        const lstmProbs = await trainAndPredictLstm(features, dynamicSeqLen, dynamicMaxEpochs);

        // ── Model 2: GRU ──────────────────────────────────────────────────
        predictLog.textContent =
          '[2/3] GRU \ud559\uc2b5 \uc911\u2026 (seq=' + dynamicSeqLen + ', EarlyStopping)';
        const gruProbs = await trainAndPredictGru(features, dynamicSeqLen, dynamicMaxEpochs);

        // ── Model 3: MLP ──────────────────────────────────────────────────
        predictLog.textContent = '[3/3] MLP \ud559\uc2b5 \uc911\u2026 (Dense 128\u219264, EarlyStopping)';
        const mlpProbs = await trainAndPredictMlp(features, Math.floor(dynamicMaxEpochs * 0.8));

        // ── Heuristics ────────────────────────────────────────────────────
        const coldHot    = coldHotWeights(numsOnly, 10);
        const gap        = gapWeights(numsOnly);
        const regression = regressionTrendWeights(numsOnly);
        const gbm        = gbmScores(numsOnly, 45);
        const [sumLow, sumHigh] = movingAverageSumBias(numsOnly, 20);

        // ── 7-way 가중 기하평균 앙상블 ────────────────────────────────────
        // LSTM 30% · GRU 25% · MLP 10% · GBM 15% · ColdHot 10% · Gap 7% · Trend 3%
        const base = new Float64Array(45);
        for (let i = 0; i < 45; i++) {
          base[i] =
            Math.pow(Math.max(lstmProbs[i],    1e-9), 0.30) *
            Math.pow(Math.max(gruProbs[i],     1e-9), 0.25) *
            Math.pow(Math.max(mlpProbs[i],     1e-9), 0.10) *
            Math.pow(Math.max(gbm[i],          1e-9), 0.15) *
            Math.pow(Math.max(coldHot[i],      1e-9), 0.10) *
            Math.pow(Math.max(gap[i],          1e-9), 0.07) *
            Math.pow(Math.max(regression[i],   1e-9), 0.03);
        }
        let s = 0;
        for (let i = 0; i < 45; i++) s += base[i];
        for (let i = 0; i < 45; i++) base[i] /= s + 1e-12;

        const sets = generateSets(Array.from(base), 5, (next * 7919) >>> 0, sumLow, sumHigh);

        const { data: ins, error } = await client
          .from('lotto_predictions')
          .insert({ target_round: next, predicted_sets: sets, actual_numbers: null, matches: null })
          .select('id').single();
        if (error) throw error;

        predictLog.textContent =
          '\ud83e\udd16 7-way \uc559\uc0c1\ube14 \uc608\uce21 \uc644\ub8cc (\ud68c\ucc28: ' + next +
          ' | ID: ' + ins.id +
          ' | \ud569\uacc4: ' + sumLow + '\u2013' + sumHigh +
          ' | LSTM\u00b730%\u00b7GRU\u00b725%\u00b7MLP\u00b710%\u00b7GBM\u00b715%\u00b7ColdHot\u00b710%\u00b7Gap\u00b77%\u00b7Trend\u00b73%)';

        const roundHeading =
          '<div class="mb-3 rounded-lg border border-amber-600/40 bg-slate-800/70 px-4 py-3 text-center">' +
          '<span class="text-xl font-bold text-amber-200">' + next + '\ud68c\ucc28</span>' +
          '<span class="ml-2 text-sm font-normal text-slate-400">\uc608\uce21 5\uc138\ud2b8 (LSTM+GRU+MLP \uc559\uc0c1\ube14)</span></div>';
        setsOut.innerHTML = roundHeading + sets.map(function (row, i) {
          return (
            '<div class="rounded-lg bg-slate-800/80 px-4 py-3 font-mono text-amber-200">Set ' +
            (i + 1) + ': ' +
            row.map(n => String(n).padStart(2, '0')).join(' ') + '</div>'
          );
        }).join('');

        await refreshStatus(client, statusEl);
      } catch (e) {
        predictLog.textContent = e.message || String(e);
        console.error(e);
      }
    };

    // ── 실제 당첨번호 입력 ─────────────────────────────────────────────────
    document.getElementById('form-result').onsubmit = async (ev) => {
      ev.preventDefault();
      const round  = Number(document.getElementById('res-round').value);
      const raw    = document.getElementById('res-nums').value.trim().split(/[\s,]+/);
      const actual = raw.map(Number).filter(n => !Number.isNaN(n));
      actual.sort((a, b) => a - b);
      const out = document.getElementById('result-out');
      out.innerHTML = '';
      if (actual.length !== 6 || new Set(actual).size !== 6) {
        out.textContent = 'Enter 6 actual winning numbers.';
        return;
      }
      try {
        const { data: predRow, error: e1 } = await client
          .from('lotto_predictions').select('*').eq('target_round', round)
          .order('id', { ascending: false }).limit(1).maybeSingle();
        if (e1) throw e1;
        if (!predRow) { out.textContent = 'No prediction row for round ' + round + '.'; return; }
        let predictedSets = predRow.predicted_sets;
        if (typeof predictedSets === 'string') predictedSets = JSON.parse(predictedSets);
        predictedSets = predictedSets.map(row => row.map(Number));
        const matches = computeMatches(predictedSets, actual);
        const { error: e2 } = await client.from('lotto_predictions')
          .update({ actual_numbers: actual, matches }).eq('id', predRow.id);
        if (e2) throw e2;
        const { error: e3 } = await client.from('lotto_history')
          .upsert({ round_no: round, numbers: actual }, { onConflict: 'round_no' });
        if (e3) throw e3;

        let html =
          '<p class="mb-2">Actual: <strong class="text-emerald-300">' +
          actual.map(n => String(n).padStart(2, '0')).join(' ') +
          '</strong></p><ul class="space-y-1 text-sm">';
        for (const m of matches) {
          const hit = m.matched_numbers.length > 0
            ? '<span class="text-amber-300">' +
              m.matched_numbers.map(n => String(n).padStart(2, '0')).join(' ') + '</span>'
            : '\u2014';
          html += '<li>Set ' + (m.set_index + 1) + ': <strong>' + m.match_count + '</strong> hit(s) \u2014 ' + hit + '</li>';
        }
        html += '</ul>';
        out.innerHTML = html;
        await refreshStatus(client, statusEl);
      } catch (e) {
        out.textContent = (e.message || String(e)) + ' \u2014 check anon key and RLS on lotto_predictions / lotto_history.';
      }
    };
  });
})();
