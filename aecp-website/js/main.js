// AECP Website JS

document.addEventListener('DOMContentLoaded', function() {

    // Theme
    var toggle = document.getElementById('themeToggle');
    var root = document.documentElement;
    var saved = localStorage.getItem('aecp-theme');
    if (saved) root.setAttribute('data-theme', saved);

    function updateIcon() {
        if (!toggle) return;
        toggle.textContent = root.getAttribute('data-theme') === 'light' ? '\u263E' : '\u263D';
    }
    updateIcon();

    if (toggle) {
        toggle.addEventListener('click', function() {
            var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
            root.setAttribute('data-theme', next);
            localStorage.setItem('aecp-theme', next);
            updateIcon();
        });
    }

    // Smooth scroll
    document.querySelectorAll('a[href^="#"]').forEach(function(a) {
        a.addEventListener('click', function(e) {
            var id = this.getAttribute('href');
            if (id === '#') return;
            var t = document.querySelector(id);
            if (t) { e.preventDefault(); t.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
        });
    });

    // Fade in
    var els = document.querySelectorAll('.fade, .grid4 .card, .grid3 .card, .grid2 .card, .ds');
    var obs = new IntersectionObserver(function(entries) {
        entries.forEach(function(e) {
            if (e.isIntersecting) { e.target.classList.add('vis'); obs.unobserve(e.target); }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -30px 0px' });
    els.forEach(function(el) { obs.observe(el); });

    // Stagger grid children
    document.querySelectorAll('.grid4, .grid3, .grid2').forEach(function(g) {
        for (var i = 0; i < g.children.length; i++) g.children[i].style.transitionDelay = (i * 60) + 'ms';
    });

    // Copy buttons
    document.querySelectorAll('pre').forEach(function(block) {
        if (block.querySelector('.cpy')) return;
        var btn = document.createElement('button');
        btn.className = 'cpy';
        btn.textContent = 'copy';
        block.appendChild(btn);
        btn.addEventListener('click', function() {
            var code = block.querySelector('code');
            if (!code) return;
            navigator.clipboard.writeText(code.textContent).then(function() {
                btn.textContent = 'copied';
                setTimeout(function() { btn.textContent = 'copy'; }, 2000);
            });
        });
    });

    // Info modals
    document.querySelectorAll('.info-trigger').forEach(function(trigger) {
        trigger.addEventListener('click', function() {
            var id = this.getAttribute('data-modal');
            var bg = document.getElementById(id);
            if (bg) bg.classList.add('open');
        });
    });

    document.querySelectorAll('.modal-bg').forEach(function(bg) {
        bg.addEventListener('click', function(e) {
            if (e.target === bg) bg.classList.remove('open');
        });
    });

    document.querySelectorAll('.modal-close').forEach(function(btn) {
        btn.addEventListener('click', function() {
            btn.closest('.modal-bg').classList.remove('open');
        });
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            document.querySelectorAll('.modal-bg.open').forEach(function(m) { m.classList.remove('open'); });
        }
    });

    // Nav shadow
    var nav = document.querySelector('.nav');
    if (nav) {
        window.addEventListener('scroll', function() {
            nav.style.boxShadow = window.scrollY > 10 ? '0 1px 3px rgba(0,0,0,.2)' : 'none';
        });
    }

});
