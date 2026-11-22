/* 犬の骨格 3D ビューア
 * データ: window.DOG_BONES（gen/build.py が作る bones_data.js）
 * 座標: データは mm・+X 頭側 / +Y 左 / +Z 上。three.js では (X, Z, -Y) に置く。
 */
(function () {
  'use strict';
  const DATA = window.DOG_BONES;
  const $ = (id) => document.getElementById(id);

  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  // ── three.js の土台 ─────────────────────────────────────────
  const canvas = $('view');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x14171c);
  const camera = new THREE.PerspectiveCamera(35, 1, 1, 10000);
  const controls = new THREE.OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.autoRotateSpeed = 1.2;

  scene.add(new THREE.HemisphereLight(0xfff6e8, 0x30353d, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 0.75);
  key.position.set(300, 800, 600);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xbfd4ff, 0.35);
  rim.position.set(-500, 300, -600);
  scene.add(rim);

  const root = new THREE.Group();
  scene.add(root);

  // 床の目印（影の代わりの円）
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(520, 64),
    new THREE.MeshBasicMaterial({ color: 0x1d2229 }));
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.5;
  root.add(floor);

  // ── 材質 ─────────────────────────────────────────────────
  const COL_BONE = 0xe8dfcb, COL_CART = 0xbcd3dc, COL_SEL = 0xff9a4d;
  function makeMat(color) {
    return new THREE.MeshStandardMaterial({ color, roughness: 0.62, metalness: 0.0 });
  }

  // ── データ → メッシュ ───────────────────────────────────────
  function b64(s, Type) {
    const bin = atob(s);
    const buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    return new Type(buf.buffer);
  }

  function geometryFor(b, mirror) {
    const vi = b64(b.v, Int16Array);
    const ti = b64(b.i, Uint16Array);
    const n = vi.length / 3;
    const pos = new Float32Array(n * 3);
    const s = mirror ? -1 : 1;
    for (let k = 0; k < n; k++) {
      const x = b.o[0] + (vi[3 * k] + 32500) * b.q;
      const y = (b.o[1] + (vi[3 * k + 1] + 32500) * b.q) * s;
      const z = b.o[2] + (vi[3 * k + 2] + 32500) * b.q;
      pos[3 * k] = x; pos[3 * k + 1] = z; pos[3 * k + 2] = -y;
    }
    let idx = ti;
    if (mirror) {             // 鏡像にすると面の向きが裏返るので巻き順を入れ替える
      idx = new Uint16Array(ti.length);
      for (let k = 0; k < ti.length; k += 3) { idx[k] = ti[k]; idx[k + 1] = ti[k + 2]; idx[k + 2] = ti[k + 1]; }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(idx, 1));
    g.computeVertexNormals();
    g.computeBoundingSphere();
    return g;
  }

  const bones = [];              // {data, meshes:[], baseColor}
  const byId = new Map();
  const pickables = [];
  for (const b of DATA.bones) {
    const isCart = b.id.startsWith('cc');
    const layer = b.layer || 'bone';
    const base = b.color ? parseInt(b.color.slice(1), 16) : (isCart ? COL_CART : COL_BONE);
    const entry = { data: b, meshes: [], baseColor: base, layer };
    const sides = b.paired ? [false, true] : [false];
    for (const mirror of sides) {
      const m = new THREE.Mesh(geometryFor(b, mirror), makeMat(entry.baseColor));
      m.userData.bone = entry;
      m.userData.side = b.paired ? (mirror ? '右' : '左') : '';
      root.add(m);
      entry.meshes.push(m);
      pickables.push(m);
    }
    bones.push(entry);
    byId.set(b.id, entry);
  }

  // 全体の中心を原点へ（床は y=0 のまま）
  const box = new THREE.Box3();
  for (const e of bones) for (const m of e.meshes) box.expandByObject(m);
  const center = box.getCenter(new THREE.Vector3());
  root.position.set(-center.x, 0, -center.z);
  scene.updateMatrixWorld(true);   // 親の移動を反映してから境界箱を測る（でないと選択時に中心がずれる）
  floor.position.x = center.x; floor.position.z = center.z;
  const sizeAll = box.getSize(new THREE.Vector3());
  const targetAll = new THREE.Vector3(0, sizeAll.y * 0.5, 0);

  // ── 一覧（ツリー） ────────────────────────────────────────
  const tree = { name: '', kids: new Map(), bones: [], items: [] };
  for (const e of bones) {
    let node = tree;
    for (const p of e.data.path) {
      if (!node.kids.has(p)) {
        const k = { name: p, kids: new Map(), bones: [], items: [], parent: node };
        node.kids.set(p, k);
        node.items.push(k);          // 骨とグループを定義順のまま並べる
      }
      node = node.kids.get(p);
    }
    node.bones.push(e);
    node.items.push(e);
  }
  function allBonesOf(node) {
    let out = [];
    for (const it of node.items) out = out.concat(it.kids ? allBonesOf(it) : [it]);
    return out;
  }
  function countMeshes(list) { return list.reduce((a, e) => a + e.meshes.length, 0); }

  const rowsByBone = new Map();
  const groupEls = [];
  function el(tag, cls, text) {
    const d = document.createElement(tag);
    if (cls) d.className = cls;
    if (text != null) d.textContent = text;
    return d;
  }
  function renderNode(node, parentEl, depth) {
    // 子グループと骨を、定義順に並べる
    for (const k of node.items) {
      if (!k.kids) { renderBone(k, parentEl); continue; }
      const g = el('div', 'grp' + (depth === 0 ? ' open' : ''));
      const row = el('div', 'row');
      row.appendChild(el('span', 'tw', '▶'));
      row.appendChild(el('span', 'nm', k.name));
      const list = allBonesOf(k);
      row.appendChild(el('span', 'n', '(' + countMeshes(list) + ')'));
      const kids = el('div', 'kids');
      g.appendChild(row); g.appendChild(kids);
      parentEl.appendChild(g);
      g._node = k;
      groupEls.push(g);
      row.addEventListener('click', (ev) => {
        if (ev.target.classList.contains('tw')) { g.classList.toggle('open'); return; }
        g.classList.add('open');
        select({ kind: 'group', node: k, label: k.name, list }, true);
        setRowSel(row);
      });
      renderNode(k, kids, depth + 1);
    }
  }
  function renderBone(e, parentEl) {
    {
      const row = el('div', 'row bone');
      row.appendChild(el('span', 'tw', ''));
      row.appendChild(el('span', 'nm', e.data.ja));
      if (e.data.src === 'scan' || e.data.src === 'derived') {
        const t = el('span', 'tag tag-' + e.data.src, e.data.src === 'scan' ? '実' : '流');
        t.title = e.data.src === 'scan' ? '実物スキャン' : '実物の形を流用';
        row.appendChild(t);
      }
      row.appendChild(el('span', 'la', e.data.la));
      row.title = e.data.ja + ' / ' + e.data.la;
      parentEl.appendChild(row);
      rowsByBone.set(e, row);
      row.addEventListener('click', () => { select({ kind: 'bone', bone: e }, true); setRowSel(row); });
    }
  }
  renderNode(tree, $('tree'), 0);

  let selRow = null;
  function setRowSel(row) {
    if (selRow) selRow.classList.remove('sel');
    selRow = row;
    if (row) row.classList.add('sel');
  }
  function revealRow(row) {
    let p = row.parentElement;
    while (p && p.id !== 'tree') { if (p.classList.contains('grp')) p.classList.add('open'); p = p.parentElement; }
    row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  // ── 検索 ────────────────────────────────────────────────
  function norm(s) { return s.toLowerCase().normalize('NFKC'); }
  function applySearch(q) {
    q = norm(q.trim());
    for (const [e, row] of rowsByBone) {
      const hit = onLayer(e) && (!q || norm(e.data.ja + ' ' + e.data.la + ' ' + e.data.path.join(' ')).includes(q));
      row.classList.toggle('hidden', !hit);
    }
    // 中身が全部隠れたグループは隠す（下から）
    for (let i = groupEls.length - 1; i >= 0; i--) {
      const g = groupEls[i];
      const any = g.querySelector(':scope > .kids > .row.bone:not(.hidden), :scope > .kids > .grp:not(.hidden)');
      g.classList.toggle('hidden', !any);
      if (q && any) g.classList.add('open');
    }
  }
  $('q').addEventListener('input', (ev) => applySearch(ev.target.value));
  $('q').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') {
      const first = [...rowsByBone.values()].find((r) => !r.classList.contains('hidden'));
      if (first) first.click();
    }
  });
  $('btn-expand').onclick = () => groupEls.forEach((g) => g.classList.add('open'));
  $('btn-collapse').onclick = () => groupEls.forEach((g) => g.classList.remove('open'));

  // ── 選択と表示 ─────────────────────────────────────────────
  let mode = 'dim';
  let layer = 'bone';            // 'bone' | 'organ' | 'both'
  let current = null;            // {kind, bone|list, label}
  const onLayer = (e) => layer === 'both' || e.layer === layer;
  function selectedSet() {
    if (!current) return null;
    return new Set(current.kind === 'bone' ? [current.bone] : current.list);
  }
  function applyLook() {
    const sel = selectedSet();
    for (const e of bones) {
      const on = sel && sel.has(e);
      const vis = onLayer(e);
      // 「両方」では骨を半透明にして、中の臓器が見えるようにする
      const ghostBone = layer === 'both' && e.layer === 'bone';
      for (const m of e.meshes) {
        const mat = m.material;
        if (e.flash) {                  // クイズの正誤の色を一時的に重ねる
          m.visible = true; mat.color.setHex(e.flash); mat.emissive.setHex(0x202020);
          mat.transparent = false; mat.opacity = 1; mat.depthWrite = true; m.renderOrder = 2;
          mat.needsUpdate = true; continue;
        }
        if (!vis && !on) {
          m.visible = false;
        } else if (!sel) {
          const a = ghostBone ? 0.22 : (e.data.alpha || 1);
          mat.color.setHex(e.baseColor); mat.emissive.setHex(0);
          mat.transparent = a < 1; mat.opacity = a;
          mat.depthWrite = a >= 1; m.visible = true;
        } else if (on) {
          // 骨はオレンジ、臓器は自分の色のまま光らせる（色で臓器を見分けられるように）
          mat.color.setHex(e.layer === 'organ' ? e.baseColor : COL_SEL);
          mat.emissive.setHex(e.layer === 'organ' ? 0x4a2a00 : 0x3a1600);
          mat.transparent = false; mat.opacity = 1;
          mat.depthWrite = true; m.visible = true;
        } else {
          mat.color.setHex(e.baseColor); mat.emissive.setHex(0);
          m.visible = mode !== 'only';
          const dim = mode === 'dim';
          mat.transparent = dim; mat.opacity = dim ? 0.13 : 1; mat.depthWrite = !dim;
        }
        m.renderOrder = on ? 1 : 0;
        mat.needsUpdate = true;
      }
    }
  }

  function boundsOf(list) {
    const b = new THREE.Box3();
    for (const e of list) for (const m of e.meshes) b.expandByObject(m);
    return b;
  }

  function select(s, fly, opts) {
    current = s;
    applyLook();
    const card = $('card');
    if (!s) { card.hidden = true; setRowSel(null); setHash(); return; }
    const list = s.kind === 'bone' ? [s.bone] : s.list;
    if (opts && opts.silent) {        // クイズ中: 名前の出るカードは出さない
      card.hidden = true;
      if (fly) flyTo(boundsOf(list), opts.dir, opts.minR);
      return;
    }
    if (s.kind === 'bone') {
      const d = s.bone.data;
      $('card-path').textContent = d.path.join(' › ');
      $('card-ja').textContent = d.ja;
      $('card-la').textContent = d.la;
      $('card-note').textContent = d.note || '';
      $('card-count').textContent = d.paired ? '左右に1対' : (d.layer === 'organ' ? '' : '正中に1個');
      $('card-src').className = 'src src-' + (d.src || 'model');
      $('card-src').innerHTML = d.src === 'scan' ? '🔬 実物スキャン <small>' + esc(d.credit || '') + '</small>'
        : d.src === 'derived' ? '🧩 実物の形を流用（' + esc(d.derived_from || '') + 'のスキャンを変形） <small>' + esc(d.credit || '') + '</small>'
        : '✏️ 模式（形は目安。実物データなし）';
      $('card-src').hidden = false;
    } else {
      const path = [];
      for (let n = s.node; n && n.name; n = n.parent) path.unshift(n.name);
      $('card-path').textContent = path.slice(0, -1).join(' › ');
      $('card-ja').textContent = s.label;
      $('card-la').textContent = '';
      const names = s.list.map((e) => e.data.ja);
      $('card-note').textContent = names.length > 14 ? names.slice(0, 14).join('・') + ' …' : names.join('・');
      $('card-count').textContent = '骨 ' + countMeshes(s.list) + ' 個';
      $('card-src').hidden = true;
    }
    card.hidden = false;
    if (fly) flyTo(boundsOf(list));
    setHash();
  }

  // ── カメラ ───────────────────────────────────────────────
  let anim = null;
  function flyTo(b, dirOverride, minR) {
    const sphere = b.getBoundingSphere(new THREE.Sphere());
    const r = Math.max(sphere.radius, minR || 18);
    const dist = r / Math.sin(fitFov() / 2) * 1.6;
    const dir = dirOverride || camera.position.clone().sub(controls.target).normalize();
    animateCam(sphere.center.clone(), sphere.center.clone().add(dir.multiplyScalar(dist)));
  }
  // 縦長の画面（スマホ）では横の画角のほうが狭いので、狭いほうで距離を決める
  function fitFov() {
    const v = camera.fov * Math.PI / 180;
    const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    return Math.min(v, 2 * Math.atan(Math.tan(v / 2) * aspect));
  }
  function animateCam(tgt, pos) {
    anim = { t: 0, from: [controls.target.clone(), camera.position.clone()], to: [tgt, pos] };
  }
  const VIEWS = {
    left: new THREE.Vector3(0, 0.05, -1),   // データの +Y（左）は three の -Z
    front: new THREE.Vector3(1, 0.1, 0),
    top: new THREE.Vector3(0.001, 1, 0.0001),
    persp: new THREE.Vector3(0.75, 0.45, -1),
  };
  function viewAll(dirName) {
    const dir = (VIEWS[dirName] || VIEWS.persp).clone().normalize();
    const b = current ? boundsOf(current.kind === 'bone' ? [current.bone] : current.list) : boundsOf(bones);
    flyTo(b, dir);
  }
  document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => viewAll(b.dataset.view)));
  document.querySelectorAll('[data-mode]').forEach((b) => b.addEventListener('click', () => {
    mode = b.dataset.mode;
    document.querySelectorAll('[data-mode]').forEach((x) => x.classList.toggle('on', x === b));
    applyLook(); setHash();
  }));
  // ── 骨格／臓器の切り替え ─────────────────────────────────
  function setLayer(ly, fly) {
    layer = ly;
    document.querySelectorAll('[data-layer]').forEach((x) => x.classList.toggle('on', x.dataset.layer === ly));
    // 一覧: 今の層に無いグループは隠す（一番上の «臓器» グループと骨のグループ）
    for (const g of groupEls) {
      if (g._node.parent !== tree) continue;
      const isOrgan = g._node.name === '臓器';
      g.classList.toggle('off-layer', !(ly === 'both' || (ly === 'organ') === isOrgan));
    }
    if (current) {
      const list = current.kind === 'bone' ? [current.bone] : current.list;
      if (!list.some(onLayer)) { select(null); fly = true; }   // 選択が外れたら全体へ引く
    }
    applyLook(); setHash();
    if (fly) flyTo(boundsOf(bones.filter(onLayer)));
  }
  document.querySelectorAll('[data-layer]').forEach((b) => b.addEventListener('click', () => setLayer(b.dataset.layer, !current)));

  // ── 一覧の開け閉め（状態は覚えておく） ─────────────────────────
  function setSide(open) {
    document.body.classList.toggle('side-closed', !open);
    $('btn-side').classList.toggle('on', open);
    try { localStorage.setItem('dogSkeleton.side', open ? '1' : '0'); } catch (e) { /* 保存できない環境 */ }
  }
  $('btn-side').onclick = () => setSide(document.body.classList.contains('side-closed'));
  {
    let saved = null;
    try { saved = localStorage.getItem('dogSkeleton.side'); } catch (e) { /* 無視 */ }
    setSide(saved === null ? window.innerWidth > 760 : saved === '1');
  }

  // ── 出典 ───────────────────────────────────────────────
  $('btn-credits').onclick = () => {
    const box = $('credits');
    const n = (k) => DATA.bones.filter((b) => (b.src || 'model') === k).length;
    box.innerHTML = '<div class="qz-head"><b>出典とデータの出どころ</b><button class="qz-x" id="cr-x" aria-label="閉じる">×</button></div>' +
      '<p class="cr-sum">🔬 実物スキャン ' + n('scan') + ' ／ 🧩 実物の形を流用 ' + n('derived') + ' ／ ✏️ 模式 ' + n('model') + '（一覧の «実» «流» の印）</p>' +
      (DATA.credits || []).map((c) => '<div class="cr"><b>' + esc(c.what) + '</b><br>' + esc(c.who) + '（' + esc(c.license) + '）<br>' +
        '<a href="' + esc(c.url) + '" target="_blank" rel="noopener">' + esc(c.url) + '</a><br><small>改変: ' + esc(c.mod) + '</small></div>').join('') +
      '<p class="cr-note">位置合わせ・形の流用・模式の骨と臓器はこのサイトで作ったものです。実物データも、骨ごとの切り分けや体内への配置は目安を含みます。</p>';
    box.hidden = false;
    $('cr-x').onclick = () => { box.hidden = true; };
  };

  // ── スマホの «⋯»（表示の設定） ───────────────────────────────
  $('btn-more').onclick = (ev) => { ev.stopPropagation(); document.body.classList.toggle('more-open'); };
  document.addEventListener('pointerdown', (ev) => {
    if (document.body.classList.contains('more-open') && !ev.target.closest('#more, #btn-more')) document.body.classList.remove('more-open');
  });
  // スマホ: 一覧で選んだら、3D が見えるようにシートを閉じる
  const isPhone = () => window.matchMedia('(max-width: 760px)').matches;
  $('tree').addEventListener('click', (ev) => {
    if (isPhone() && ev.target.closest('.row.bone')) setTimeout(() => setSide(false), 150);
  });

  $('btn-all').onclick = () => { select(null); flyTo(boundsOf(bones), VIEWS.persp.clone().normalize()); };
  $('card-close').onclick = () => select(null);
  $('spin').addEventListener('change', (ev) => { controls.autoRotate = ev.target.checked; });

  // ── マウス: ホバー表示とクリック選択 ───────────────────────────
  const ray = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  function pick(ev) {
    const r = canvas.getBoundingClientRect();
    mouse.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(mouse, camera);
    const hits = ray.intersectObjects(pickables.filter((m) => m.visible && (!m.material.transparent || (!current && m.material.opacity > 0.3))), false);
    if (!hits.length) return null;
    // 半透明（肺など）の奥に不透明なものがあれば、そちらを選ぶ（肺越しに心臓を押せるように）
    const solid = hits.find((h) => !h.object.material.transparent);
    return (solid || hits[0]).object;
  }
  const hover = $('hover');
  canvas.addEventListener('pointermove', (ev) => {
    if (ev.buttons) { hover.hidden = true; return; }
    const m = pick(ev);
    if (!m) { hover.hidden = true; canvas.style.cursor = ''; return; }
    if (window.DogQuiz && window.DogQuiz.active) { hover.hidden = true; canvas.style.cursor = 'pointer'; return; }
    const d = m.userData.bone.data;
    hover.textContent = d.ja + (m.userData.side ? '（' + m.userData.side + '）' : '');
    const r = canvas.getBoundingClientRect();
    hover.style.left = (ev.clientX - r.left + 14) + 'px';
    hover.style.top = (ev.clientY - r.top + 10) + 'px';
    hover.hidden = false;
    canvas.style.cursor = 'pointer';
  });
  canvas.addEventListener('pointerleave', () => { hover.hidden = true; });
  let down = null;
  canvas.addEventListener('pointerdown', (ev) => { down = [ev.clientX, ev.clientY]; });
  canvas.addEventListener('pointerup', (ev) => {
    if (!down || Math.hypot(ev.clientX - down[0], ev.clientY - down[1]) > 4) return;
    const m = pick(ev);
    if (window.DogQuiz && window.DogQuiz.active) { window.DogQuiz.onPick(m); return; }
    if (!m) { if (current) select(null); return; }
    const e = m.userData.bone;
    select({ kind: 'bone', bone: e }, false);
    const row = rowsByBone.get(e);
    $('q').value = ''; applySearch('');
    setRowSel(row); revealRow(row);
  });

  // ── URL の # で状態を共有（#sel=humerus&mode=only&view=left） ─────────
  let hashLock = false;
  function setHash() {
    if (hashLock) return;
    const p = new URLSearchParams();
    if (current) p.set('sel', current.kind === 'bone' ? current.bone.data.id : 'g:' + current.label);
    if (mode !== 'dim') p.set('mode', mode);
    if (layer !== 'bone') p.set('layer', layer);
    const h = p.toString();
    history.replaceState(null, '', h ? '#' + h : location.pathname + location.search);
  }
  function fromHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    hashLock = true;
    const ly = p.get('layer');
    if (ly) setLayer(ly, false);
    const m = p.get('mode');
    if (m) document.querySelector('[data-mode="' + m + '"]')?.click();
    const s = p.get('sel');
    if (s && s.startsWith('g:')) {
      const name = s.slice(2);
      const g = groupEls.find((x) => x._node.name === name);
      if (g) g.querySelector(':scope > .row .nm').click();
    } else if (s && byId.has(s)) {
      rowsByBone.get(byId.get(s)).click();
      revealRow(rowsByBone.get(byId.get(s)));
    }
    if (p.get('view')) viewAll(p.get('view'));
    if (anim) {                    // 開いた直後はアニメーションせずにその視点へ
      controls.target.copy(anim.to[0]); camera.position.copy(anim.to[1]); anim = null;
    }
    hashLock = false;
    setHash();
  }

  // ── 名前ラベル（3D の上に浮かべる） ───────────────────────────
  // 選んでいるものがあればそれに、無ければ «表示中の大きめの骨・臓器» に付ける。
  let labelsOn = false;
  const labelBox = $('labels');
  const labelEls = new Map();      // entry -> div
  const tmpV = new THREE.Vector3();
  // ラベルの対象: {key, text, meshes}。meshes の中心（左右はカメラに近いほう）に付ける。
  function labelTargets() {
    if (window.DogQuiz && window.DogQuiz.active) return [];
    const one = (e) => ({ key: e, text: e.data.ja, sets: [e.meshes] });
    if (current) return (current.kind === 'bone' ? [current.bone] : current.list).filter(onLayer).slice(0, 80).map(one);
    // 何も選んでいないとき: 4個以上並ぶグループ（胸椎・肋骨など）はグループ名ひとつにまとめる
    const out = [];
    const walk = (node) => {
      for (const it of node.items) {
        if (it.kids) {
          const list = allBonesOf(it).filter(onLayer);
          const leaf = it.items.every((x) => !x.kids);
          if (leaf && list.length >= 4 && it.name !== '臓器') {
            out.push({ key: it, text: it.name, sets: [list.map((e) => e.meshes[0]), list.map((e) => e.meshes[e.meshes.length - 1])] });
          } else walk(it);
        } else if (onLayer(it) && it.radius >= (it.layer === 'organ' ? 4 : 12)) out.push(one(it));
      }
    };
    walk(tree);
    return out;
  }
  function setLabels(on) {
    labelsOn = on;
    $('btn-labels').classList.toggle('on', on);
    if (!on) { labelBox.innerHTML = ''; labelEls.clear(); }
  }
  $('btn-labels').onclick = () => setLabels(!labelsOn);
  for (const e of bones) {
    e.radius = e.meshes[0].geometry.boundingSphere.radius;
  }
  function updateLabels() {
    if (!labelsOn) return;
    const targets = labelTargets();
    const want = new Set(targets.map((t) => t.key));
    for (const [k, d] of labelEls) if (!want.has(k)) { d.remove(); labelEls.delete(k); }
    const w = canvas.clientWidth, h = canvas.clientHeight;
    for (const t of targets) {
      let d = labelEls.get(t.key);
      if (!d) {
        d = document.createElement('div');
        d.className = 'lbl' + (t.key.kids ? ' grp' : '');
        d.textContent = t.text;
        labelBox.appendChild(d);
        labelEls.set(t.key, d);
      }
      // 左右のある骨は、カメラに近いほうに付ける
      let best = null, bd = Infinity;
      for (const ms of t.sets) {
        const c = new THREE.Vector3();
        for (const m of ms) c.add(tmpV.copy(m.geometry.boundingSphere.center).applyMatrix4(m.matrixWorld));
        c.multiplyScalar(1 / ms.length);
        const dd = c.distanceTo(camera.position);
        if (dd < bd) { bd = dd; best = c; }
      }
      best.project(camera);
      const vis = best.z < 1 && Math.abs(best.x) < 1.05 && Math.abs(best.y) < 1.05;
      d.style.display = vis ? '' : 'none';
      if (vis) d.style.transform = 'translate(' + ((best.x + 1) / 2 * w).toFixed(1) + 'px,' + ((1 - best.y) / 2 * h).toFixed(1) + 'px)';
    }
  }

  // ── 描画ループ ─────────────────────────────────────────────
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
  }
  function loop() {
    resize();
    if (anim) {
      anim.t = Math.min(1, anim.t + 0.06);
      const e = anim.t * anim.t * (3 - 2 * anim.t);
      controls.target.lerpVectors(anim.from[0], anim.to[0], e);
      camera.position.lerpVectors(anim.from[1], anim.to[1], e);
      if (anim.t >= 1) anim = null;
    }
    controls.update();
    renderer.render(scene, camera);
    updateLabels();
    requestAnimationFrame(loop);
  }

  // 初期視点: 斜め前から全体
  const d0 = VIEWS.persp.clone().normalize();
  const all = boundsOf(bones).getBoundingSphere(new THREE.Sphere());
  controls.target.copy(all.center);
  resize();
  camera.position.copy(all.center).add(d0.multiplyScalar(all.radius / Math.sin(fitFov() / 2) * 0.72));
  $('loading').remove();
  // 最初の表示にも層を反映（しないと骨格モードで臓器が見え、一覧にも臓器が出る）。
  // このとき URL の # を書き換えないこと（先に消すと #sel=… の共有が効かなくなる）。
  hashLock = true; setLayer(layer, false); hashLock = false;
  fromHash();
  loop();
  function flash(list, hex) {
    for (const e of bones) e.flash = 0;
    for (const e of list) e.flash = hex;
    applyLook();
  }
  window.__dog = {
    select, viewAll, byId, bones, tree, rowsByBone, flyTo, boundsOf, applyLook, flash, setSide,
    setLayer, getLayer: () => layer, setMode: (m) => document.querySelector('[data-mode="' + m + '"]').click(),
    getMode: () => mode, VIEWS, setLabels: (on) => setLabels(on),
  };
  document.dispatchEvent(new Event('dog-ready'));
  // ホーム画面に追加したとき、オフラインでも開けるように（https のときだけ）
  if ('serviceWorker' in navigator && location.protocol === 'https:') navigator.serviceWorker.register('sw.js').catch(() => {});
})();
