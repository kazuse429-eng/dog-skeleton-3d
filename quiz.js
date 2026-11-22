/* 犬の骨格・臓器 3D — クイズと学習記録
 *
 * 形式:
 *   what  「これは何？」光った骨・臓器の名前を4択で答える
 *   where 「どこ？」出された名前の骨・臓器を3Dでクリックする
 * 記録: localStorage 'dogSkeleton.stats' に {id: {c: 正解数, w: 不正解数, last: 1|0, t: 時刻}}
 */
(function () {
  'use strict';

  function start() {
    const D = window.__dog;
    const $ = (id) => document.getElementById(id);
    const panel = $('quiz');
    const KEY = 'dogSkeleton.stats';

    // ── 記録 ───────────────────────────────────────────────
    let stats = {};
    try { stats = JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { stats = {}; }
    function save() { try { localStorage.setItem(KEY, JSON.stringify(stats)); } catch (e) { /* 保存できない環境 */ } }
    function record(e, ok) {
      const s = stats[e.data.id] || { c: 0, w: 0, last: 0, t: 0 };
      if (ok) s.c++; else s.w++;
      s.last = ok ? 1 : 0;
      s.t = Date.now();
      stats[e.data.id] = s;
      save();
      paintRow(e);
    }
    function status(e) {
      const s = stats[e.data.id];
      if (!s) return '';
      if (s.last === 1 && s.c >= 2 && s.c > s.w) return 'ok';
      if (s.last === 0 || s.w >= s.c) return 'weak';
      return 'mid';
    }
    function paintRow(e) {
      const row = D.rowsByBone.get(e);
      if (!row) return;
      let dot = row.querySelector('.st');
      if (!dot) {
        dot = document.createElement('span');
        dot.className = 'st';
        row.insertBefore(dot, row.children[1]);
      }
      const st = status(e);
      dot.className = 'st' + (st ? ' st-' + st : '');
      dot.title = st === 'ok' ? '覚えた' : st === 'weak' ? '苦手' : st === 'mid' ? '練習中' : '';
    }
    D.bones.forEach(paintRow);

    // ── 出題範囲（一覧のグループから作る） ─────────────────────────
    const ranges = [];
    function collect(node) {
      let out = [];
      for (const it of node.items) out = out.concat(it.kids ? collect(it) : [it]);
      return out;
    }
    const boneAll = D.bones.filter((e) => e.layer === 'bone');
    const organAll = D.bones.filter((e) => e.layer === 'organ');
    ranges.push({ key: 'bone', label: '骨：全部', list: boneAll, layer: 'bone' });
    for (const k of D.tree.items) {
      if (!k.kids || k.name === '臓器') continue;
      ranges.push({ key: 'b:' + k.name, label: '骨：' + k.name, list: collect(k), layer: 'bone' });
    }
    ranges.push({ key: 'organ', label: '臓器：全部', list: organAll, layer: 'organ' });
    const organRoot = D.tree.items.find((k) => k.kids && k.name === '臓器');
    if (organRoot) {
      for (const k of organRoot.items) {
        if (k.kids) ranges.push({ key: 'o:' + k.name, label: '臓器：' + k.name, list: collect(k), layer: 'organ' });
      }
    }
    ranges.push({ key: 'all', label: '骨と臓器：全部', list: D.bones.slice(), layer: 'both' });

    const isTiny = (e) => e.data.path.some((p) => p === '指骨' || p === '趾骨');

    // ── 設定（前回の選択を覚えておく） ─────────────────────────────
    let cfg = { range: 'bone', type: 'what', lang: 'ja', n: 10, weak: true, tiny: false };
    try { Object.assign(cfg, JSON.parse(localStorage.getItem('dogSkeleton.quizcfg') || '{}')); } catch (e) { /* 無視 */ }
    function saveCfg() { try { localStorage.setItem('dogSkeleton.quizcfg', JSON.stringify(cfg)); } catch (e) { /* 無視 */ } }

    // ── 状態 ───────────────────────────────────────────────
    const Q = { active: false, onPick, debug: () => run };   // debug はテスト用
    window.DogQuiz = Q;
    let run = null;      // {range, items, i, score, wrong:[], answered, tries}
    let restore = null;  // クイズ前の表示状態

    const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
    const nameOf = (e) => (cfg.lang === 'la' ? e.data.la : e.data.ja);
    const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

    function pool(range) {
      return range.list.filter((e) => cfg.tiny || !isTiny(e));
    }
    function progressOf(list) {
      let ok = 0, weak = 0, seen = 0;
      for (const e of list) {
        const st = status(e);
        if (st) seen++;
        if (st === 'ok') ok++;
        if (st === 'weak') weak++;
      }
      return { ok, weak, seen, total: list.length };
    }

    // ── 画面: 設定 ───────────────────────────────────────────
    function showSetup() {
      const r = ranges.find((x) => x.key === cfg.range) || ranges[0];
      const pr = progressOf(pool(r));
      const weakList = pool(r).filter((e) => status(e) === 'weak').slice(0, 6).map((e) => e.data.ja);
      panel.innerHTML = `
        <div class="qz-head"><b>🎯 クイズ</b><button class="qz-x" data-act="close" aria-label="閉じる">×</button></div>
        <label class="qz-f">範囲
          <select id="qz-range">${ranges.map((x) => `<option value="${x.key}"${x.key === r.key ? ' selected' : ''}>${esc(x.label)}（${pool(x).length}）</option>`).join('')}</select></label>
        <div class="qz-f">形式
          <span class="qz-seg">
            <button data-type="what" class="${cfg.type === 'what' ? 'on' : ''}">これは何？（4択）</button>
            <button data-type="where" class="${cfg.type === 'where' ? 'on' : ''}">どこ？（クリック）</button>
          </span></div>
        <div class="qz-f">名前
          <span class="qz-seg">
            <button data-lang="ja" class="${cfg.lang === 'ja' ? 'on' : ''}">和名</button>
            <button data-lang="la" class="${cfg.lang === 'la' ? 'on' : ''}">ラテン名</button>
          </span></div>
        <div class="qz-f">問題数
          <span class="qz-seg">${[10, 20, 0].map((n) => `<button data-n="${n}" class="${cfg.n === n ? 'on' : ''}">${n || '全部'}</button>`).join('')}</span></div>
        <label class="qz-c"><input type="checkbox" id="qz-weak"${cfg.weak ? ' checked' : ''}> 苦手・まだのものを優先</label>
        <label class="qz-c"><input type="checkbox" id="qz-tiny"${cfg.tiny ? ' checked' : ''}> 指・趾の骨（指骨）も出す</label>
        <div class="qz-prog">
          <div class="bar"><i style="width:${pr.total ? pr.ok / pr.total * 100 : 0}%"></i><s style="width:${pr.total ? pr.weak / pr.total * 100 : 0}%"></s></div>
          覚えた <b>${pr.ok}</b> ／ 苦手 <b class="weak">${pr.weak}</b> ／ 全 ${pr.total}
          ${weakList.length ? `<div class="qz-weaklist">苦手: ${weakList.map(esc).join('・')}</div>` : ''}
        </div>
        <div class="qz-btns"><button class="qz-go" data-act="go">はじめる</button>
          <button class="qz-sub" data-act="reset">記録を消す</button></div>`;
      panel.hidden = false;
      $('qz-range').onchange = (ev) => { cfg.range = ev.target.value; saveCfg(); showSetup(); };
      $('qz-weak').onchange = (ev) => { cfg.weak = ev.target.checked; saveCfg(); };
      $('qz-tiny').onchange = (ev) => { cfg.tiny = ev.target.checked; saveCfg(); showSetup(); };
      panel.querySelectorAll('[data-type]').forEach((b) => b.onclick = () => { cfg.type = b.dataset.type; saveCfg(); showSetup(); });
      panel.querySelectorAll('[data-lang]').forEach((b) => b.onclick = () => { cfg.lang = b.dataset.lang; saveCfg(); showSetup(); });
      panel.querySelectorAll('[data-n]').forEach((b) => b.onclick = () => { cfg.n = +b.dataset.n; saveCfg(); showSetup(); });
      panel.querySelector('[data-act=go]').onclick = () => begin(r);
      panel.querySelector('[data-act=close]').onclick = close;
      panel.querySelector('[data-act=reset]').onclick = () => {
        if (!confirm('クイズの記録（覚えた・苦手）をすべて消しますか？')) return;
        stats = {}; save(); D.bones.forEach(paintRow); showSetup();
      };
    }

    // ── 進行 ───────────────────────────────────────────────
    function enter() {
      if (!restore) {
        restore = { layer: D.getLayer(), mode: D.getMode(), side: !document.body.classList.contains('side-closed') };
      }
      Q.active = true;
      document.body.classList.add('quiz-on');
      D.setSide(false);                       // 一覧は答えが見えてしまうので閉じる
      D.setLabels(false);
    }
    function close() {
      Q.active = false;
      run = null;
      panel.hidden = true;
      document.body.classList.remove('quiz-on');
      D.flash([], 0);
      D.select(null);
      if (restore) {
        D.setLayer(restore.layer, false);
        D.setMode(restore.mode);
        D.setSide(restore.side);
        restore = null;
      }
    }

    function pickItems(range, only) {
      let items = only ? only.slice() : pool(range);
      shuffle(items);
      if (cfg.weak && !only) {
        // 苦手 → まだ → 練習中 → 覚えた の順（同じ順位の中はランダム）
        const rank = { weak: 0, '': 1, mid: 2, ok: 3 };
        items.sort((a, b) => rank[status(a)] - rank[status(b)]);
      }
      const n = cfg.n || items.length;
      return items.slice(0, n);
    }

    function begin(range, only) {
      enter();
      run = { range, items: pickItems(range, only), i: 0, score: 0, wrong: [], answered: false, tries: 0 };
      if (!run.items.length) { showSetup(); return; }
      D.setLayer(range.layer === 'both' ? 'both' : range.layer, false);
      D.setMode('dim');
      ask();
    }

    function head() {
      return `<div class="qz-head"><b>${run.i + 1} / ${run.items.length}</b>
        <span class="qz-score">正解 ${run.score}</span>
        <button class="qz-x" data-act="quit" aria-label="やめる">×</button></div>`;
    }
    function wireQuit() { panel.querySelector('[data-act=quit]').onclick = () => { if (run.i > 0 || run.answered) result(); else close(); }; }

    function choicesFor(target) {
      const label = nameOf(target);
      const same = (e) => nameOf(e) === label;
      // 近い仲間（同じグループ）から先に選ぶほうが勉強になる
      const parentPath = target.data.path.join('/');
      const sib = shuffle(run.range.list.filter((e) => e !== target && !same(e) && e.data.path.join('/') === parentPath));
      const near = shuffle(run.range.list.filter((e) => e !== target && !same(e) && e.data.path[0] === target.data.path[0]));
      const far = shuffle(D.bones.filter((e) => e !== target && !same(e) && e.layer === target.layer));
      const out = [];
      const used = new Set([label]);
      for (const e of sib.concat(near, far)) {
        const nm = nameOf(e);
        if (used.has(nm)) continue;
        used.add(nm); out.push(e);
        if (out.length === 3) break;
      }
      return shuffle([target].concat(out));
    }

    function ask() {
      const t = run.items[run.i];
      run.answered = false; run.tries = 0;
      D.flash([], 0);
      if (cfg.type === 'what') {
        // 小さい骨でも周りが見える距離で止める（どこの骨か分かるように）
        D.select({ kind: 'bone', bone: t }, true, { silent: true, minR: t.layer === 'organ' ? 60 : 70 });
        const ch = choicesFor(t);
        panel.innerHTML = head() + `
          <div class="qz-q">光っているのは？</div>
          <div class="qz-ch">${ch.map((e, k) => `<button data-k="${k}"><small>${k + 1}</small>${esc(nameOf(e))}</button>`).join('')}</div>
          <div class="qz-fb" id="qz-fb"></div>`;
        panel.querySelectorAll('[data-k]').forEach((b) => b.onclick = () => answerWhat(ch[+b.dataset.k], b, ch));
      } else {
        D.select(null);
        const b = D.boundsOf(run.range.list.filter((e) => e.layer === t.layer || run.range.layer === 'both'));
        D.flyTo(b);
        panel.innerHTML = head() + `
          <div class="qz-q">「<b>${esc(nameOf(t))}</b>」はどれ？ <small>3Dでクリック</small></div>
          <div class="qz-fb" id="qz-fb"></div>
          <div class="qz-btns"><button class="qz-sub" data-act="giveup">わからない</button></div>`;
        panel.querySelector('[data-act=giveup]').onclick = () => revealWhere(false);
      }
      wireQuit();
    }

    function feedback(ok, t, extra) {
      const fb = $('qz-fb');
      fb.className = 'qz-fb ' + (ok ? 'ok' : 'ng');
      fb.innerHTML = `<div class="qz-mark">${ok ? '⭕ 正解！' : '❌ ざんねん'}</div>
        <div class="qz-ans"><b>${esc(t.data.ja)}</b> <i>${esc(t.data.la)}</i></div>
        ${extra || ''}<div class="qz-note">${esc(t.data.note || '')}</div>
        <div class="qz-btns"><button class="qz-go" data-act="next">${run.i + 1 < run.items.length ? '次へ ▶' : '結果を見る'}</button></div>`;
      fb.querySelector('[data-act=next]').onclick = next;
      fb.querySelector('[data-act=next]').focus();
    }

    function answerWhat(e, btn, ch) {
      if (run.answered) return;
      run.answered = true;
      const t = run.items[run.i];
      const ok = e === t;
      if (ok) run.score++; else run.wrong.push(t);
      record(t, ok);
      panel.querySelectorAll('[data-k]').forEach((b) => {
        const c = ch[+b.dataset.k];
        b.disabled = true;
        if (c === t) b.classList.add('right');
        else if (b === btn) b.classList.add('wrong');
      });
      if (!ok) D.flash([e], 0x6b7fd7);    // 選んだほう（間違い）も青で見せて違いが分かるように
      feedback(ok, t, ok ? '' : `<div class="qz-sub2">選んだのは「${esc(nameOf(e))}」（青）</div>`);
      setTimeout(() => D.flash([], 0), ok ? 0 : 1600);
    }

    function onPick(m) {
      if (!run || cfg.type !== 'where' || run.answered) return true;
      const t = run.items[run.i];
      if (!m) return true;
      const e = m.userData.bone;
      if (e === t) {
        run.answered = true;
        if (run.tries === 0) run.score++; else run.wrong.push(t);
        record(t, run.tries === 0);
        D.flash([t], 0x3cc46a);
        feedback(run.tries === 0, t, run.tries ? '<div class="qz-sub2">2回目で正解（記録は «苦手»）</div>' : '');
        return true;
      }
      run.tries++;
      D.flash([e], 0xe2504a);
      const fb = $('qz-fb');
      fb.className = 'qz-fb ng';
      fb.innerHTML = `<div>それは「${esc(nameOf(e))}」。${run.tries >= 2 ? '' : 'もう一回！'}</div>`;
      setTimeout(() => { if (!run.answered) D.flash([], 0); }, 900);
      if (run.tries >= 2) setTimeout(() => revealWhere(false), 700);
      return true;
    }

    function revealWhere() {
      if (run.answered) return;
      run.answered = true;
      const t = run.items[run.i];
      run.wrong.push(t);
      record(t, false);
      D.flash([], 0);
      D.select({ kind: 'bone', bone: t }, true, { silent: true });
      feedback(false, t, '<div class="qz-sub2">オレンジが正解の場所</div>');
    }

    function next() {
      D.flash([], 0);
      run.i++;
      if (run.i >= run.items.length) result(); else ask();
    }

    function result() {
      const n = run.answered ? run.i + 1 : run.i;
      const pct = n ? Math.round(run.score / n * 100) : 0;
      const wrong = [...new Set(run.wrong)];
      D.select(null);
      panel.innerHTML = `
        <div class="qz-head"><b>結果</b><button class="qz-x" data-act="close" aria-label="閉じる">×</button></div>
        <div class="qz-res"><span class="big">${run.score}</span> / ${n} 問 正解（${pct}%）</div>
        ${wrong.length ? `<div class="qz-wlist">間違えたもの（押すと場所を表示）:
          ${wrong.map((e, k) => `<button data-w="${k}">${esc(e.data.ja)}</button>`).join('')}</div>` : '<div class="qz-res">全問正解！🎉</div>'}
        <div class="qz-btns">
          ${wrong.length ? '<button class="qz-go" data-act="retry-wrong">間違えたものだけもう一度</button>' : ''}
          <button class="qz-sub" data-act="again">もう一度</button>
          <button class="qz-sub" data-act="setup">設定にもどる</button></div>`;
      const range = run.range;
      panel.querySelectorAll('[data-w]').forEach((b) => b.onclick = () => {
        D.select({ kind: 'bone', bone: wrong[+b.dataset.w] }, true, { silent: true });
      });
      panel.querySelector('[data-act=close]').onclick = close;
      panel.querySelector('[data-act=again]').onclick = () => begin(range);
      panel.querySelector('[data-act=setup]').onclick = () => { run = null; showSetup(); };
      const rw = panel.querySelector('[data-act=retry-wrong]');
      if (rw) rw.onclick = () => begin(range, wrong);
    }

    // ── キーボード（1〜4 で選ぶ、Enter で次、Esc でやめる） ──────────────
    document.addEventListener('keydown', (ev) => {
      if (panel.hidden || ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT') return;
      if (ev.key === 'Escape') { close(); return; }
      if (!run) return;
      if (/^[1-4]$/.test(ev.key) && cfg.type === 'what' && !run.answered) {
        const b = panel.querySelector('[data-k="' + (+ev.key - 1) + '"]');
        if (b) b.click();
      } else if (ev.key === 'Enter' && run.answered) {
        const b = panel.querySelector('[data-act=next]');
        if (b) { ev.preventDefault(); b.click(); }
      }
    });

    $('btn-quiz').onclick = () => {
      if (!panel.hidden) { close(); return; }
      enter();
      showSetup();
    };
  }

  if (window.__dog) start(); else document.addEventListener('dog-ready', start);
})();
