(function () {
  'use strict';

  const MIN_FOR_FIRST_PREDICT = 5;
  const LSTM_EPOCHS = 60; // fallback (동적 계산으로 덮어씀)

  function getClient() {
    const c = window.LOTTO_CONFIG;
    if (!c || !c.supabaseUrl || !c.supabaseAnonKey) {
      throw new Error('Set supabaseUrl and supabaseAnonKey in web/js/config.js');
    }
    const lib = window.supabase;
    if (!lib || typeof lib.createClient !== 'function') {
      throw new Error('Supabase library failed to load');
    }
    return lib.createClient(c.supabaseUrl, c.supabaseAnonKey);
  }

  function mulberry32(a) {
    return function () {
      let t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function roundsToMultihot(rounds) {
    const x = [];
    for (let i = 0; i < rounds.length; i++) {
      const row = new Float32Array(45);
      for (const n of rounds[i]) row[n - 1] = 1;
      x.push(row);
    }
    return x;
  }

  function buildSequences(multihot, seqLen) {
    const xs = [];
    const ys = [];
    for (let t = seqLen; t < multihot.length; t++) {
      const chunk = [];
      for (let s = 0; s < seqLen; s++) {
        chunk.push(Array.from(multihot[t - seqLen + s]));
      }
      xs.push(chunk);
      ys.push(Array.from(multihot[t]));
    }
    return { xs, ys };
  }

  function coldHotWeights(historyRounds, windowN) {
    const counts = new Float64Array(45);
    const recent =
      historyRounds.length >= windowN
        ? historyRounds.slice(-windowN)
        : historyRounds;
    for (const nums of recent) {
      for (const n of nums) counts[n - 1] += 1;
    }
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

  function patternMatchBoost(historyRounds, seqLen) {
    if (historyRounds.length <= seqLen + 1) {
      return new Float64Array(45).fill(1);
    }
    const mh = roundsToMultihot(historyRounds);
    const rows = mh.map((r) => Array.from(r));
    function meanWindow(start, len) {
      const acc = new Float64Array(45);
      for (let i = 0; i < len; i++) {
        for (let j = 0; j < 45; j++) acc[j] += rows[start + i][j];
      }
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
      let wnorm = 0;
      for (let j = 0; j < 45; j++) wnorm += win[j] * win[j];
      wnorm = Math.sqrt(wnorm) + 1e-9;
      let dot = 0;
      for (let j = 0; j < 45; j++) dot += win[j] * target[j];
      const sim = dot / (wnorm * tnorm);
      if (sim > 0.92) {
        const nxt = rows[t + 1];
        for (let j = 0; j < 45; j++) boost[j] += nxt[j] * (sim - 0.92) * 10;
      }
    }
    let mx = 0;
    for (let j = 0; j < 45; j++) mx = Math.max(mx, boost[j]);
    const out = new Float64Array(45);
    if (mx > 0) {
      for (let j = 0; j < 45; j++) out[j] = 1 + 0.6 * (boost[j] / (mx + 1e-9));
    } else {
      out.fill(1);
    }
    return out;
  }

  // ── NEW: Gap weights (미출현 기간 분석) ───────────────────────────────────
  function gapWeights(historyRounds, size) {
    const lastSeen = new Array(size).fill(historyRounds.length);
    for (let i = 0; i < size; i++) {
      for (let r = historyRounds.length - 1; r >= 0; r--) {
        if (historyRounds[r].includes(i + 1)) {
          lastSeen[i] = historyRounds.length - 1 - r;
          break;
        }
      }
    }
    const maxGap = Math.max(...lastSeen) + 1;
    const w = new Float64Array(size);
    for (let i = 0; i < size; i++) {
      const norm = lastSeen[i] / maxGap;
      w[i] = 0.4 + 0.9 * norm;
    }
    return w;
  }

  // ── NEW: Regression trend weights (번호별 최근 트렌드 상승/하락) ──────────
  function regressionTrendWeights(historyRounds, window_, size) {
    window_ = window_ || 20;
    size = size || 45;
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

  // ── NEW: Simplified Gradient Boost (XGBoost-style frequency model) ───────
  function gbmScores(historyRounds, size) {
    size = size || 45;
    const n = historyRounds.length;
    const scores = new Float64Array(size);
    const window5  = historyRounds.slice(-5);
    const window10 = historyRounds.slice(-10);
    const window30 = historyRounds.slice(-30);
    const allRounds = historyRounds;
    function cnt(slice, idx) {
      let c = 0;
      for (const r of slice) if (r.includes(idx + 1)) c++;
      return c;
    }
    for (let i = 0; i < size; i++) {
      const f5  = cnt(window5,  i) / (window5.length  + 1e-9);
      const f10 = cnt(window10, i) / (window10.length + 1e-9);
      const f30 = cnt(window30, i) / (window30.length + 1e-9);
      const fall = cnt(allRounds, i) / (n + 1e-9);
      // weighted residual boosting: recent windows have higher leaf weight
      scores[i] = 0.35 * f5 + 0.30 * f10 + 0.20 * f30 + 0.15 * fall + 1e-6;
    }
    return scores;
  }

  function movingAverageSumBias(historyRounds, lookback) {
    const slice = historyRounds.slice(-lookback);
    const sums = slice.map((r) => r.reduce((a, b) => a + b, 0));
    if (sums.length < 3) return [100, 175];
    const mu = sums.reduce((a, b) => a + b, 0) / sums.length;
    const var_ =
      sums.reduce((a, b) => a + (b - mu) * (b - mu), 0) / sums.length;
    const sigma = Math.sqrt(var_) + 1e-6;
    return [
      Math.max(75, Math.floor(mu - 1.2 * sigma)),
      Math.min(220, Math.floor(mu + 1.2 * sigma)),
    ];
  }

  function oddEvenOk(nums) {
    const odd = nums.filter((n) => n % 2 === 1).length;
    const even = 6 - odd;
    return (
      (odd === 2 && even === 4) ||
      (odd === 3 && even === 3) ||
      (odd === 4 && even === 2)
    );
  }

  // ── NEW: consecutive number check ────────────────────────────────────────
  function hasExcessiveConsecutive(nums) {
    const sorted = nums.slice().sort(function (a, b) { return a - b; });
    let run = 1;
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i] === sorted[i - 1] + 1) {
        run++;
        if (run > 2) return true; // 3개 이상 연속 → 제외
      } else {
        run = 1;
      }
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

  function weightedPickSix(prob45, rng) {
    const picked = [];
    const p = prob45.slice();
    for (let k = 0; k < 6; k++) {
      let s = 0;
      const avail = [];
      for (let i = 0; i < 45; i++) {
        if (!picked.includes(i)) {
          s += p[i];
          avail.push(i);
        }
      }
      if (s <= 0 || avail.length === 0) break;
      let r = rng() * s;
      let chosen = avail[0];
      for (const i of avail) {
        r -= p[i];
        if (r <= 0) {
          chosen = i;
          break;
        }
      }
      picked.push(chosen);
    }
    return picked;
  }

  function sampleSixFromScores(scores, rng, sumLow, sumHigh) {
    const p = scores.map((x) => Math.max(x, 1e-12));
    const sumP = p.reduce((a, b) => a + b, 0);
    for (let i = 0; i < 45; i++) p[i] /= sumP;

    function draw(strictOE) {
      for (let tries = 0; tries < 8000; tries++) {
        const picked = weightedPickSix(p, rng);
        if (picked.length < 6) continue;
        const nums = picked.map((i) => i + 1).sort((a, b) => a - b);
        if (passesFilters(nums, sumLow, sumHigh, strictOE)) return nums;
      }
      return null;
    }
    let r = draw(true);
    if (r) return r;
    r = draw(false);
    if (r) return r;
    const idx = p
      .map((v, i) => [v, i])
      .sort((a, b) => b[0] - a[0])
      .slice(0, 12)
      .map((x) => x[1]);
    for (let tries = 0; tries < 8000; tries++) {
      const sh = idx.slice().sort(() => rng() - 0.5);
      const pick = sh.slice(0, 6).sort((a, b) => a - b);
      const nums = pick.map((i) => i + 1);
      if (passesFilters(nums, sumLow, sumHigh, false)) return nums;
    }
    return null;
  }

  function generateSets(baseScores, numSets, seed, sumLow, sumHigh) {
    const rng = mulberry32(seed >>> 0);
    const sets = [];
    const used = new Set();
    let attempt = 0;
    while (sets.length < numSets && attempt < numSets * 400) {
      attempt++;
      const scores = baseScores.map((b) => {
        const noise = Math.exp((rng() - 0.5) * 0.14);
        return Math.max(b * noise, 1e-12);
      });
      const got = sampleSixFromScores(scores, rng, sumLow, sumHigh);
      if (!got) continue;
      const key = got.join(',');
      if (used.has(key)) continue;
      used.add(key);
      sets.push(got);
    }
    while (sets.length < numSets) {
      const pool = [];
      while (pool.length < 6) {
        const n = 1 + Math.floor(rng() * 45);
        if (!pool.includes(n)) pool.push(n);
      }
      pool.sort((a, b) => a - b);
      const key = pool.join(',');
      if (!used.has(key)) {
        used.add(key);
        sets.push(pool);
      }
    }
    return sets;
  }

  async function trainAndPredictLstm(multihot, seqLen, epochs) {
    const { xs, ys } = buildSequences(multihot, seqLen);
    if (xs.length === 0) throw new Error('No training samples');
    const n = xs.length;
    const xArr = new Float32Array(n * seqLen * 45);
    const yArr = new Float32Array(n * 45);
    for (let i = 0; i < n; i++) {
      for (let s = 0; s < seqLen; s++) {
        for (let d = 0; d < 45; d++) {
          xArr[i * seqLen * 45 + s * 45 + d] = xs[i][s][d];
        }
      }
      for (let d = 0; d < 45; d++) yArr[i * 45 + d] = ys[i][d];
    }
    const xsT = tf.tensor3d(xArr, [n, seqLen, 45]);
    const ysT = tf.tensor2d(yArr, [n, 45]);

    const model = tf.sequential();
    model.add(
      tf.layers.lstm({
        units: 128,
        inputShape: [seqLen, 45],
        returnSequences: false,
      })
    );
    model.add(tf.layers.dense({ units: 45, activation: 'sigmoid' }));
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'binaryCrossentropy',
    });

    await model.fit(xsT, ysT, {
      epochs,
      batchSize: Math.min(16, n),
      verbose: 0,
      shuffle: true,
    });

    const last = [];
    for (let s = 0; s < seqLen; s++) {
      for (let d = 0; d < 45; d++) {
        last.push(multihot[multihot.length - seqLen + s][d]);
      }
    }
    const lastT = tf.tensor3d(last, [1, seqLen, 45]);
    const pred = model.predict(lastT);
    const probs = await pred.data();
    pred.dispose();
    lastT.dispose();
    xsT.dispose();
    ysT.dispose();
    model.dispose();

    return Array.from(probs);
  }

  async function fetchHistory(client) {
    const { data, error } = await client
      .from('lotto_history')
      .select('round_no, numbers')
      .order('round_no', { ascending: true });
    if (error) throw error;
    return (data || []).map((r) => ({
      round: r.round_no,
      numbers: (r.numbers || []).map(Number).sort((a, b) => a - b),
    }));
  }

  function computeMatches(predictedSets, actual) {
    const aset = new Set(actual);
    return predictedSets.map((s, i) => {
      const matched = s.filter((n) => aset.has(n)).sort((a, b) => a - b);
      return {
        set_index: i,
        predicted: s,
        matched_numbers: matched,
        match_count: matched.length,
      };
    });
  }

  async function refreshStatus(client, el) {
    const rows = await fetchHistory(client);
    const maxR = rows.length ? Math.max(...rows.map((r) => r.round)) : null;
    const next = maxR == null ? 1 : maxR + 1;
    const maxLabel = maxR == null ? '\u2014' : String(maxR);
    el.innerHTML =
      '<p>\uc800\uc7a5\ub41c \ud68c\ucc28: <strong>' +
      rows.length +
      '</strong>\uac1c</p><p>\ucd5c\ub300 \ud68c\ucc28: <strong>' +
      maxLabel +
      '</strong> \u2192 \ub2e4\uc74c \uc608\uce21 \ub300\uc0c1: <strong>' +
      next +
      '\ud68c</strong></p>';
    const resRoundInput = document.getElementById('res-round');
    if (resRoundInput) {
      resRoundInput.value = String(next);
    }
    return { rows, maxR, next };
  }

  document.addEventListener('DOMContentLoaded', async () => {
    const statusEl = document.getElementById('status');
    const logEl = document.getElementById('log');
    const predictLog = document.getElementById('predict-log');
    const setsOut = document.getElementById('sets-out');
    let client;

    try {
      client = getClient();
    } catch (e) {
      statusEl.innerHTML =
        '<p class="text-red-400">' + (e.message || String(e)) + '</p>';
      return;
    }

    document.getElementById('btn-refresh').onclick = async () => {
      try {
        await refreshStatus(client, statusEl);
        logEl.textContent = '';
      } catch (e) {
        logEl.textContent = e.message || String(e);
      }
    };

    await refreshStatus(client, statusEl);

    document.getElementById('form-input').onsubmit = async (ev) => {
      ev.preventDefault();
      const round = Number(document.getElementById('in-round').value);
      const raw = document.getElementById('in-nums')
        .value.trim()
        .split(/[\s,]+/);
      const nums = raw.map(Number).filter((n) => !Number.isNaN(n));
      nums.sort((a, b) => a - b);
      if (nums.length !== 6 || new Set(nums).size !== 6) {
        logEl.textContent = 'Enter exactly 6 distinct numbers.';
        return;
      }
      if (nums.some((n) => n < 1 || n > 45)) {
        logEl.textContent = 'Numbers must be 1–45.';
        return;
      }
      try {
        const { error } = await client
          .from('lotto_history')
          .upsert(
            { round_no: round, numbers: nums },
            { onConflict: 'round_no' }
          );
        if (error) throw error;
        logEl.textContent = 'Saved round ' + round + ': ' + nums.join(', ');
        await refreshStatus(client, statusEl);
      } catch (e) {
        logEl.textContent =
          (e.message || String(e)) +
          ' — check anon key and RLS policies on lotto_history.';
      }
    };

    document.getElementById('btn-predict').onclick = async () => {
      setsOut.innerHTML = '';
      predictLog.textContent = '';
      try {
        const { rows, maxR, next } = await refreshStatus(client, statusEl);
        if (!rows.length) {
          predictLog.textContent = '\ud68c\ucc28 \ub370\uc774\ud130\ub97c \uba3c\uc800 \uc785\ub825\ud558\uc138\uc694.';
          return;
        }
        if (rows.length < MIN_FOR_FIRST_PREDICT) {
          predictLog.textContent =
            '\uc608\uce21\ud558\ub824\uba74 \ucd5c\uc18c ' +
            MIN_FOR_FIRST_PREDICT +
            '\ud68c\ucc28 \uc774\uc0c1 \uc785\ub825\uc774 \ud544\uc694\ud569\ub2c8\ub2e4. (\ud604\uc7ac ' +
            rows.length +
            '\ud68c\ucc28)';
          return;
        }
        const numsOnly = rows.map((r) => r.numbers);
        if (numsOnly.length < 2) {
          predictLog.textContent = '\uc608\uce21\ud558\ub824\uba74 \ucd5c\uc18c 2\ud68c\ucc28 \uc774\uc0c1 \ud544\uc694\ud569\ub2c8\ub2e4.';
          return;
        }

        let seqLen = Math.min(12, numsOnly.length - 1);
        seqLen = Math.max(1, seqLen);

        // 회차 수에 따라 epoch·seqLen 동적 조정 (많을수록 빠르게)
        // 데이터가 많으면 샘플도 많아 적은 epoch로도 수렴
        const n = numsOnly.length;
        let dynamicEpochs;
        let dynamicSeqLen;
        if (n <= 30) {
          dynamicEpochs = 80;  dynamicSeqLen = Math.min(8,  seqLen);
        } else if (n <= 100) {
          dynamicEpochs = 60;  dynamicSeqLen = Math.min(10, seqLen);
        } else if (n <= 200) {
          dynamicEpochs = 40;  dynamicSeqLen = Math.min(12, seqLen);
        } else if (n <= 350) {
          dynamicEpochs = 25;  dynamicSeqLen = Math.min(12, seqLen);
        } else {
          dynamicEpochs = 18;  dynamicSeqLen = Math.min(10, seqLen);
        }

        predictLog.textContent =
          'LSTM \ud559\uc2b5 \uc911\u2026 (' + n + '\ud68c\ucc28 \ub370\uc774\ud130, ' +
          dynamicEpochs + ' epoch, seq=' + dynamicSeqLen +
          ') \u2192 5-way \uc559\uc0c1\ube14 \uc801\uc6a9 \uc911';
        const multihot = roundsToMultihot(numsOnly);
        const lstmProbs = await trainAndPredictLstm(
          multihot,
          dynamicSeqLen,
          dynamicEpochs
        );

        const coldHot   = coldHotWeights(numsOnly, 10);                // Cold/Hot
        const pat       = patternMatchBoost(numsOnly, dynamicSeqLen);  // 패턴 매칭
        const gap       = gapWeights(numsOnly);                         // 미출현 Gap
        const regression = regressionTrendWeights(numsOnly);           // 회귀 트렌드
        const gbm       = gbmScores(numsOnly);                          // GBM 빈도 모델
        const [sumLow, sumHigh] = movingAverageSumBias(numsOnly, 20);

        // ── 5-way 앙상블 (기하 평균 방식으로 과도한 지배 방지) ─────────────
        // LSTM 40% · GBM 20% · Cold/Hot 15% · Gap 15% · Regression 10%
        const base = new Float64Array(45);
        for (let i = 0; i < 45; i++) {
          const v =
            Math.pow(Math.max(lstmProbs[i], 1e-9), 0.40) *
            Math.pow(Math.max(gbm[i],        1e-9), 0.20) *
            Math.pow(Math.max(coldHot[i],    1e-9), 0.15) *
            Math.pow(Math.max(gap[i],        1e-9), 0.15) *
            Math.pow(Math.max(regression[i], 1e-9), 0.10);
          base[i] = v;
        }
        let s = 0;
        for (let i = 0; i < 45; i++) s += base[i];
        for (let i = 0; i < 45; i++) base[i] /= s + 1e-12;

        const sets = generateSets(
          Array.from(base),
          5,
          (next * 7919) >>> 0,
          sumLow,
          sumHigh
        );

        const { data: ins, error } = await client
          .from('lotto_predictions')
          .insert({
            target_round: next,
            predicted_sets: sets,
            actual_numbers: null,
            matches: null,
          })
          .select('id')
          .single();
        if (error) throw error;

        predictLog.textContent =
          '\ud83e\udd16 \uc559\uc0c1\ube14 \uc608\uce21 \uc644\ub8cc (\ud68c\ucc28: ' +
          next +
          ' | \uc800\uc7a5 ID: ' +
          ins.id +
          ' | \ud569\uacc4 \ubc94\uc704: ' +
          sumLow + '\u2013' + sumHigh +
          ' | LSTM\u00b740%\u00b7GBM\u00b720%\u00b7Cold\u2215Hot\u00b715%\u00b7Gap\u00b715%\u00b7Trend\u00b710%)';
        const roundHeading =
          '<div class="mb-3 rounded-lg border border-amber-600/40 bg-slate-800/70 px-4 py-3 text-center">' +
          '<span class="text-xl font-bold text-amber-200">' +
          next +
          '\ud68c\ucc28</span>' +
          '<span class="ml-2 text-sm font-normal text-slate-400">\uc608\uce21 5\uc138\ud2b8</span></div>';
        setsOut.innerHTML =
          roundHeading +
          sets
            .map(function (row, i) {
              return (
                '<div class="rounded-lg bg-slate-800/80 px-4 py-3 font-mono text-amber-200">Set ' +
                (i + 1) +
                ': ' +
                row.map((n) => String(n).padStart(2, '0')).join(' ') +
                '</div>'
              );
            })
            .join('');
        await refreshStatus(client, statusEl);
      } catch (e) {
        predictLog.textContent = e.message || String(e);
        console.error(e);
      }
    };

    document.getElementById('form-result').onsubmit = async (ev) => {
      ev.preventDefault();
      const round = Number(document.getElementById('res-round').value);
      const raw = document
        .getElementById('res-nums')
        .value.trim()
        .split(/[\s,]+/);
      const actual = raw.map(Number).filter((n) => !Number.isNaN(n));
      actual.sort((a, b) => a - b);
      const out = document.getElementById('result-out');
      out.innerHTML = '';
      if (actual.length !== 6 || new Set(actual).size !== 6) {
        out.textContent = 'Enter 6 actual winning numbers.';
        return;
      }
      try {
        const { data: predRow, error: e1 } = await client
          .from('lotto_predictions')
          .select('*')
          .eq('target_round', round)
          .order('id', { ascending: false })
          .limit(1)
          .maybeSingle();
        if (e1) throw e1;
        if (!predRow) {
          out.textContent = 'No prediction row for round ' + round + '.';
          return;
        }
        let predictedSets = predRow.predicted_sets;
        if (typeof predictedSets === 'string') {
          predictedSets = JSON.parse(predictedSets);
        }
        predictedSets = predictedSets.map((row) => row.map(Number));
        const matches = computeMatches(predictedSets, actual);
        const { error: e2 } = await client
          .from('lotto_predictions')
          .update({ actual_numbers: actual, matches })
          .eq('id', predRow.id);
        if (e2) throw e2;
        const { error: e3 } = await client
          .from('lotto_history')
          .upsert(
            { round_no: round, numbers: actual },
            { onConflict: 'round_no' }
          );
        if (e3) throw e3;

        let html =
          '<p class="mb-2">Actual: <strong class="text-emerald-300">' +
          actual.map((n) => String(n).padStart(2, '0')).join(' ') +
          '</strong></p><ul class="space-y-1 text-sm">';
        for (const m of matches) {
          const hit =
            m.matched_numbers.length > 0
              ? '<span class="text-amber-300">' +
                m.matched_numbers
                  .map((n) => String(n).padStart(2, '0'))
                  .join(' ') +
                '</span>'
              : '—';
          html +=
            '<li>Set ' +
            (m.set_index + 1) +
            ': <strong>' +
            m.match_count +
            '</strong> hit(s) — ' +
            hit +
            '</li>';
        }
        html += '</ul>';
        out.innerHTML = html;
        await refreshStatus(client, statusEl);
      } catch (e) {
        out.textContent =
          (e.message || String(e)) +
          ' — check anon key and RLS on lotto_predictions / lotto_history.';
      }
    };
  });
})();
