// Section tracking for integrations page
document.addEventListener('DOMContentLoaded', function() {
    var sections = document.querySelectorAll('.section[id]');
    var tocLinks = document.querySelectorAll('.toc a');
    if (!sections.length || !tocLinks.length) return;

    var observer = new IntersectionObserver(function(entries) {
        entries.forEach(function(entry) {
            if (entry.isIntersecting) {
                var id = entry.target.getAttribute('id');
                tocLinks.forEach(function(link) {
                    link.classList.toggle('active', link.getAttribute('href') === '#' + id);
                });
            }
        });
    }, { threshold: 0.2, rootMargin: '-80px 0px -60% 0px' });

    sections.forEach(function(s) { observer.observe(s); });
});
