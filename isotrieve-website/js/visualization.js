/* Point-cloud alignment — signature hero visual
   Two embedding spaces: source (violet) misaligned → map applied → target (cyan) */

(function () {
  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  ready(function () {
    var canvas = document.getElementById('transferViz');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var N = 36;
    var points = [];
    var phase = 0; // 0 scatter → 1 aligning → 2 settled
    var t = 0;
    var running = true;
    var prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function cssVar(name, fallback) {
      var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
      return v || fallback;
    }

    function mulberry32(a) {
      return function () {
        a |= 0; a = (a + 0x6d2b79f5) | 0;
        var t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    }

    function initPoints() {
      var rng = mulberry32(42);
      points = [];
      for (var i = 0; i < N; i++) {
        // Source cloud: elongated / rotated cluster (left space geometry)
        var sx = (rng() - 0.5) * 1.6;
        var sy = (rng() - 0.5) * 0.9;
        var ang = 0.55;
        var rx = sx * Math.cos(ang) - sy * Math.sin(ang);
        var ry = sx * Math.sin(ang) + sy * Math.cos(ang);

        // Target: same semantic neighbors, different geometry (scale + shear)
        var tx = rx * 0.85 + ry * 0.35 + (rng() - 0.5) * 0.08;
        var ty = -rx * 0.25 + ry * 1.05 + (rng() - 0.5) * 0.08;

        // Misaligned start: source points in wrong orientation relative to target
        var mx = rx * 1.1 + 0.15;
        var my = ry * 0.7 - 0.1;

        points.push({
          mx: mx, my: my,
          tx: tx, ty: ty,
          x: mx, y: my,
          pulse: rng()
        });
      }
    }

    function resize() {
      var rect = canvas.parentElement.getBoundingClientRect();
      var w = Math.max(rect.width, 200);
      var h = Math.max(rect.height, 200);
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      canvas.style.width = w + 'px';
      canvas.style.height = h + 'px';
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function easeInOut(u) {
      return u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
    }

    function project(px, py, w, h) {
      var pad = 48;
      var cx = w / 2;
      var cy = h / 2 - 8;
      var scale = Math.min(w, h) * 0.32;
      return { x: cx + px * scale, y: cy + py * scale, pad: pad, scale: scale, cx: cx, cy: cy };
    }

    function drawAxes(w, h, color, alpha) {
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 5]);
      var midY = h / 2 - 8;
      var midX = w / 2;
      ctx.beginPath();
      ctx.moveTo(40, midY);
      ctx.lineTo(w - 40, midY);
      ctx.moveTo(midX, 36);
      ctx.lineTo(midX, h - 52);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }

    function drawGhostTargets(w, h, alpha) {
      var cyan = cssVar('--cyan', '#4ecdc4');
      ctx.save();
      ctx.globalAlpha = alpha;
      for (var i = 0; i < points.length; i++) {
        var p = points[i];
        var q = project(p.tx, p.ty, w, h);
        ctx.beginPath();
        ctx.arc(q.x, q.y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = cyan;
        ctx.fill();
      }
      ctx.restore();
    }

    function draw() {
      var w = canvas.width / dpr;
      var h = canvas.height / dpr;
      var violet = cssVar('--violet', '#8b7ec8');
      var cyan = cssVar('--cyan', '#4ecdc4');
      var accent = cssVar('--accent', '#e8a04a');
      var border = cssVar('--border', '#2a2a3d');
      var bg = cssVar('--surface', '#14141e');

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);

      // subtle grid
      ctx.save();
      ctx.strokeStyle = border;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1;
      for (var gx = 0; gx < w; gx += 28) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, h); ctx.stroke();
      }
      for (var gy = 0; gy < h; gy += 28) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
      }
      ctx.restore();

      drawAxes(w, h, border, 0.5);

      // Ghost target constellation while aligning / after
      var ghostA = phase === 0 ? 0.15 : (phase === 1 ? 0.25 + t * 0.2 : 0.35);
      drawGhostTargets(w, h, ghostA);

      // Progress of linear map
      var u = 0;
      if (phase === 1) u = easeInOut(t);
      else if (phase === 2) u = 1;

      for (var i = 0; i < points.length; i++) {
        var p = points[i];
        p.x = p.mx + (p.tx - p.mx) * u;
        p.y = p.my + (p.ty - p.my) * u;

        var q = project(p.x, p.y, w, h);
        var r = 3.2 + (phase === 2 ? Math.sin(Date.now() / 600 + p.pulse * 6) * 0.4 : 0);

        // trail during align
        if (phase === 1 && u > 0.05 && u < 0.95) {
          var from = project(p.mx, p.my, w, h);
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(q.x, q.y);
          ctx.strokeStyle = accent;
          ctx.globalAlpha = 0.2;
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.globalAlpha = 1;
        }

        var fill = u < 0.5
          ? violet
          : (u < 1 ? accent : cyan);

        ctx.beginPath();
        ctx.arc(q.x, q.y, r, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.globalAlpha = 0.9;
        ctx.fill();

        if (u > 0.85) {
          ctx.beginPath();
          ctx.arc(q.x, q.y, r + 3, 0, Math.PI * 2);
          ctx.strokeStyle = cyan;
          ctx.globalAlpha = (u - 0.85) / 0.15 * 0.35;
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }

      // Center transform glyph during align
      if (phase === 1) {
        ctx.save();
        ctx.globalAlpha = 0.55 + t * 0.35;
        ctx.fillStyle = accent;
        ctx.font = '600 11px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('W · x', w / 2, 28);
        ctx.restore();
      }

      if (phase === 2) {
        ctx.save();
        ctx.globalAlpha = 0.7;
        ctx.fillStyle = cyan;
        ctx.font = '600 11px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('aligned', w / 2, 28);
        ctx.restore();
      }
    }

    function tick(now) {
      if (!running) return;

      if (prefersReduced) {
        phase = 2;
        t = 1;
        draw();
        return;
      }

      if (phase === 0) {
        t += 0.008;
        if (t >= 1) { phase = 1; t = 0; }
      } else if (phase === 1) {
        t += 0.0065;
        if (t >= 1) { phase = 2; t = 1; }
      } else {
        // settled — gentle idle, then replay
        t += 0.0015;
        if (t > 4) {
          phase = 0;
          t = 0;
          // slight reseed of misalignment noise for replay interest
          for (var i = 0; i < points.length; i++) {
            var p = points[i];
            p.mx = p.tx * 0.3 + (Math.random() - 0.5) * 1.4;
            p.my = p.ty * 0.2 + (Math.random() - 0.5) * 1.1;
          }
        }
      }

      draw();
      requestAnimationFrame(tick);
    }

    initPoints();
    resize();
    window.addEventListener('resize', function () {
      resize();
      draw();
    });

    // Replay when theme changes (colors)
    var obs = new MutationObserver(function () { draw(); });
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

    if (prefersReduced) {
      phase = 2;
      t = 1;
      draw();
    } else {
      requestAnimationFrame(tick);
    }
  });
})();
