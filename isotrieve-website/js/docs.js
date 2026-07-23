// Docs page JS

document.addEventListener('DOMContentLoaded', function() {
    var sections = document.querySelectorAll('.ds');
    var links = document.querySelectorAll('.docs-side a');

    // Active nav on scroll
    function updateNav() {
        var pos = window.scrollY + 80;
        var cur = '';
        sections.forEach(function(s) { if (pos >= s.offsetTop) cur = s.id; });
        links.forEach(function(l) { l.classList.toggle('on', l.getAttribute('href') === '#' + cur); });
    }
    window.addEventListener('scroll', updateNav);
    updateNav();

    // Sidebar smooth scroll
    links.forEach(function(l) {
        l.addEventListener('click', function(e) {
            e.preventDefault();
            var t = document.getElementById(this.getAttribute('href').substring(1));
            if (t) window.scrollTo({ top: t.offsetTop - 56, behavior: 'smooth' });
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

    // Heading anchors
    document.querySelectorAll('.ds h2, .ds h3').forEach(function(h) {
        if (!h.id) h.id = h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
        var a = document.createElement('a');
        a.className = 'ha';
        a.href = '#' + h.id;
        a.textContent = '#';
        h.appendChild(a);
    });
});
