// Docs page JS

document.addEventListener('DOMContentLoaded', function() {
    var sections = document.querySelectorAll('.ds');
    var links = document.querySelectorAll('.docs-side a');

    function navOffset() {
        var h = getComputedStyle(document.documentElement).getPropertyValue('--nav-h').trim();
        var px = h.endsWith('rem') ? parseFloat(h) * 16 : parseFloat(h);
        return (px || 56) + 8;
    }

    function docTop(el) {
        return el.getBoundingClientRect().top + window.scrollY;
    }

    // Active nav on scroll — keyed off section ids (unchanged)
    function updateNav() {
        var pos = window.scrollY + navOffset() + 24;
        var cur = '';
        sections.forEach(function(s) {
            if (s.id && pos >= docTop(s)) cur = s.id;
        });
        links.forEach(function(l) {
            l.classList.toggle('on', l.getAttribute('href') === '#' + cur);
        });
    }
    window.addEventListener('scroll', updateNav, { passive: true });
    updateNav();

    // Sidebar smooth scroll
    links.forEach(function(l) {
        l.addEventListener('click', function(e) {
            e.preventDefault();
            var id = this.getAttribute('href').substring(1);
            var t = document.getElementById(id);
            if (t) window.scrollTo({ top: docTop(t) - navOffset(), behavior: 'smooth' });
        });
    });

    // Search
    var input = document.getElementById('docsSearch');
    if (input) {
        input.addEventListener('input', function() {
            var q = this.value.toLowerCase();
            links.forEach(function(l) {
                var show = q === '' || l.textContent.toLowerCase().indexOf(q) !== -1;
                l.style.display = show ? '' : 'none';
            });
        });
    }

    // Heading anchors (section titles in .docs-head; overview still uses .docs-prose)
    document.querySelectorAll('.docs-head h2, .docs-head h3, .docs-prose h2, .docs-prose h3').forEach(function(h) {
        if (!h.id) h.id = h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
        if (h.querySelector('.ha')) return;
        var a = document.createElement('a');
        a.className = 'ha';
        a.href = '#' + h.id;
        a.textContent = '#';
        h.appendChild(a);
    });
});
