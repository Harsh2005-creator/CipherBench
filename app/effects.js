/* CipherBench — presentation-only behaviour (injected once by streamlit_app.py).
   Scroll reveals, sliding nav indicator, mobile menu, cursor spotlight,
   magnetic buttons, scroll progress and a sparse ambient node field.
   Touches no data; every effect degrades to a static page if this never runs. */
// NOTE: st.html parses this file as HTML, so never write a literal "<" followed by a letter in here (build nodes with createElement).
(function () {
  if (window.__cipherbench) return;
  window.__cipherbench = true;

  var doc = document;
  var root = doc.documentElement;
  root.classList.add('cb-js');
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- reveal system ---------- */
  var io = 'IntersectionObserver' in window ? new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -6% 0px' }) : null;

  function watch(el) {
    if (el.__cbw) return;
    el.__cbw = true;
    if (!io) { el.classList.add('in'); return; }
    io.observe(el);
  }
  function add(el, cls) { cls.split(' ').forEach(function (c) { el.classList.add(c); }); watch(el); }
  function holder(el) { return el.closest('[data-testid="stElementContainer"]') || el; }

  var RULES = [
    ['.page-head', 'reveal reveal-blur', 0],
    ['.story-head', 'reveal reveal-up', 0],
    ['.vs-block', 'reveal reveal-up', 0],
    ['.big330', 'reveal reveal-blur', 0],
    ['.cipher-field', 'reveal reveal-left', 0],
    ['.eco-wrap', 'reveal reveal-up', 0],
    ['.verdict', 'reveal reveal-scale', 0],
    ['.flow, .feats, .models, .go-grid, .facts', 'reveal-stagger', 0],
    ['.exp-grid', '', 0],
    ['[data-testid="stPlotlyChart"]', 'reveal reveal-scale sk', 0],
    ['[data-testid="stDataFrame"]', 'reveal reveal-up', 0],
    ['.card, [data-testid="stMetric"], .insight-box, .diagram', 'reveal reveal-up', 1]
  ];

  function tag(scope) {
    RULES.forEach(function (r) {
      scope.querySelectorAll(r[0]).forEach(function (el) {
        var t = r[2] ? holder(el) : el;
        if (t.__cbw) return;
        if (r[1]) add(t, r[1]); else watch(t);
        if (t.classList.contains('reveal-stagger')) {
          Array.prototype.forEach.call(t.children, function (c, i) { c.style.setProperty('--k', i); });
        }
        var col = el.closest('[data-testid="stColumn"]');
        if (col && col.parentElement) {
          t.style.setProperty('--i', Array.prototype.indexOf.call(col.parentElement.children, col));
        }
      });
    });
  }

  /* ---------- navbar: sliding indicator, scrolled state, hamburger ---------- */
  function navbar() { return doc.querySelector('.st-key-navbar'); }

  function placeIndicator() {
    var nav = navbar();
    if (!nav) return;
    var group = nav.querySelector('[data-testid="stRadioGroup"]');
    if (!group) return;
    var ind = group.querySelector('.nav-ind');
    if (!ind) { ind = doc.createElement('div'); ind.className = 'nav-ind'; group.appendChild(ind); }
    var sel = group.querySelector('label[data-selected="true"]');
    if (!sel || group.offsetParent === null || getComputedStyle(group).flexDirection === 'column') {
      ind.classList.remove('on'); root.classList.remove('cb-has-ind'); return;
    }
    ind.style.width = sel.offsetWidth + 'px';
    ind.style.height = sel.offsetHeight + 'px';
    ind.style.transform = 'translate(' + sel.offsetLeft + 'px,' + sel.offsetTop + 'px)';
    ind.classList.add('on'); root.classList.add('cb-has-ind');
  }

  doc.addEventListener('click', function (e) {
    var t = e.target;
    var nav = navbar();
    var tog = t.closest && t.closest('.nav-toggle');
    if (tog) { root.classList.toggle('cb-nav-open'); return; }
    var go = t.closest && t.closest('[data-go]');
    if (go) {
      e.preventDefault();
      var want = go.getAttribute('data-go');
      var labels = doc.querySelectorAll('.st-key-navbar label[data-testid="stRadioOption"]');
      for (var i = 0; i < labels.length; i++) {
        if (labels[i].textContent.trim() === want) { labels[i].click(); break; }
      }
      var main = doc.querySelector('[data-testid="stMain"]');
      if (main) main.scrollTo({ top: 0, behavior: 'smooth' });
    }
  });
  /* capture phase: react-aria may stop propagation of the radio's own click */
  doc.addEventListener('click', function (e) {
    var t = e.target;
    if (root.classList.contains('cb-nav-open') && t.closest &&
        (t.closest('label[data-testid="stRadioOption"]') || !t.closest('.st-key-navbar'))) {
      setTimeout(function () { root.classList.remove('cb-nav-open'); }, 120);
    }
  }, true);
  doc.addEventListener('keydown', function (e) {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.matches && e.target.matches('[data-go]')) { e.preventDefault(); e.target.click(); }
    if (e.key === 'Escape') root.classList.remove('cb-nav-open');
  });

  /* ---------- scroll: progress bar + navbar state (capture: Streamlit scrolls an inner container) ---------- */
  var bar = doc.createElement('div');
  bar.id = 'cb-progress'; bar.appendChild(doc.createElement('i'));
  doc.body.appendChild(bar);
  var barI = bar.firstChild;
  doc.addEventListener('scroll', function (e) {
    var s = e.target === doc ? doc.scrollingElement : e.target;
    if (!s || s.scrollHeight === undefined) return;
    var max = s.scrollHeight - s.clientHeight;
    if (max > 200) barI.style.transform = 'scaleX(' + Math.min(1, s.scrollTop / max) + ')';
    root.classList.toggle('cb-scrolled', s.scrollTop > 12);
  }, true);

  /* ---------- pointer: card spotlight + magnetic buttons ---------- */
  var SPOT = '.card, .model-card, .go-card, [data-testid="stMetric"], [data-testid="stPlotlyChart"]';
  doc.addEventListener('pointermove', function (e) {
    var el = e.target.closest && e.target.closest(SPOT);
    if (el) {
      var r = el.getBoundingClientRect();
      el.style.setProperty('--mx', (e.clientX - r.left) + 'px');
      el.style.setProperty('--my', (e.clientY - r.top) + 'px');
    }
    if (reduce) return;
    var b = e.target.closest && e.target.closest('.stButton > button');
    if (b) {
      var br = b.getBoundingClientRect();
      b.style.setProperty('--tx', ((e.clientX - br.left - br.width / 2) * 0.08).toFixed(1) + 'px');
      b.style.setProperty('--ty', ((e.clientY - br.top - br.height / 2) * 0.16).toFixed(1) + 'px');
    }
  }, { passive: true });
  doc.addEventListener('pointerout', function (e) {
    var b = e.target.closest && e.target.closest('.stButton > button');
    if (b && !b.contains(e.relatedTarget)) { b.style.setProperty('--tx', '0px'); b.style.setProperty('--ty', '0px'); }
  });

  /* ---------- ambient node field: sparse, slow, paused when hidden ---------- */
  function ambient() {
    if (reduce) return;
    var c = doc.createElement('canvas');
    c.id = 'cb-bg';
    doc.body.appendChild(c);
    var x = c.getContext('2d');
    var w = 0, h = 0, dpr = Math.min(window.devicePixelRatio || 1, 2), pts = [];
    function size() {
      w = window.innerWidth; h = window.innerHeight;
      c.width = w * dpr; c.height = h * dpr; c.style.width = w + 'px'; c.style.height = h + 'px';
      x.setTransform(dpr, 0, 0, dpr, 0, 0);
      var n = Math.round(Math.min(46, Math.max(16, w / 34)));
      pts = [];
      for (var i = 0; i < n; i++) pts.push({ x: Math.random() * w, y: Math.random() * h, vx: (Math.random() - .5) * .12, vy: (Math.random() - .5) * .12, r: Math.random() * 1.3 + .5, hue: Math.random() });
    }
    size();
    window.addEventListener('resize', size);
    function frame() {
      if (!doc.hidden) {
        x.clearRect(0, 0, w, h);
        for (var i = 0; i < pts.length; i++) {
          var p = pts[i];
          p.x += p.vx; p.y += p.vy;
          if (p.x < -10) p.x = w + 10; if (p.x > w + 10) p.x = -10;
          if (p.y < -10) p.y = h + 10; if (p.y > h + 10) p.y = -10;
          for (var j = i + 1; j < pts.length; j++) {
            var q = pts[j], dx = p.x - q.x, dy = p.y - q.y, d = dx * dx + dy * dy;
            if (d < 19600) { x.strokeStyle = 'rgba(140,125,255,' + (0.09 * (1 - d / 19600)).toFixed(3) + ')'; x.lineWidth = 1; x.beginPath(); x.moveTo(p.x, p.y); x.lineTo(q.x, q.y); x.stroke(); }
          }
          x.fillStyle = p.hue > .8 ? 'rgba(249,115,22,.5)' : p.hue > .6 ? 'rgba(34,211,165,.42)' : 'rgba(150,135,255,.5)';
          x.beginPath(); x.arc(p.x, p.y, p.r, 0, 6.2832); x.fill();
        }
      }
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }
  ambient();

  /* ---------- keep everything in sync with Streamlit re-renders ---------- */
  var queued = false;
  function sync() {
    queued = false;
    tag(doc);
    placeIndicator();
  }
  new MutationObserver(function () {
    if (!queued) { queued = true; requestAnimationFrame(sync); }
  }).observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['data-selected'] });
  window.addEventListener('resize', placeIndicator);
  sync();
  setTimeout(placeIndicator, 400);
  if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(placeIndicator);
})();
