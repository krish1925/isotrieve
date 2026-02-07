// Documentation page JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Update active nav link on scroll
    const sections = document.querySelectorAll('.docs-section');
    const navLinks = document.querySelectorAll('.nav-link');
    
    function updateActiveNav() {
        let current = '';
        const scrollPos = window.scrollY + 150;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    }
    
    window.addEventListener('scroll', updateActiveNav);
    updateActiveNav();
    
    // Smooth scroll for nav links
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            const targetSection = document.getElementById(targetId);
            
            if (targetSection) {
                const offsetTop = targetSection.offsetTop - 80;
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Search functionality
    const searchInput = document.getElementById('docsSearch');
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            
            navLinks.forEach(link => {
                const text = link.textContent.toLowerCase();
                const parentSection = link.closest('.nav-section');
                
                if (query === '' || text.includes(query)) {
                    link.style.display = 'block';
                    if (parentSection) {
                        parentSection.style.display = 'block';
                    }
                } else {
                    link.style.display = 'none';
                }
            });
            
            // Also search in section content
            sections.forEach(section => {
                const text = section.textContent.toLowerCase();
                const sectionId = section.getAttribute('id');
                const correspondingLink = document.querySelector(`a[href="#${sectionId}"]`);
                
                if (query !== '' && text.includes(query) && correspondingLink) {
                    correspondingLink.style.display = 'block';
                    const parentSection = correspondingLink.closest('.nav-section');
                    if (parentSection) {
                        parentSection.style.display = 'block';
                    }
                }
            });
        });
    }
    
    // Add copy buttons to code blocks
    const codeBlocks = document.querySelectorAll('.code-block');
    codeBlocks.forEach(block => {
        const wrapper = document.createElement('div');
        wrapper.style.position = 'relative';
        block.parentNode.insertBefore(wrapper, block);
        wrapper.appendChild(block);
        
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-code-btn';
        copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
        copyBtn.title = 'Copy code';
        
        copyBtn.addEventListener('click', function() {
            const code = block.querySelector('code').textContent;
            navigator.clipboard.writeText(code).then(() => {
                copyBtn.innerHTML = '<i class="fas fa-check"></i>';
                copyBtn.style.background = '#10b981';
                setTimeout(() => {
                    copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
                    copyBtn.style.background = '';
                }, 2000);
            });
        });
        
        wrapper.appendChild(copyBtn);
    });
    
    // Add anchor links to headings
    const headings = document.querySelectorAll('.docs-section h2, .docs-section h3');
    headings.forEach(heading => {
        if (!heading.id) {
            heading.id = heading.textContent.toLowerCase()
                .replace(/[^a-z0-9]+/g, '-')
                .replace(/(^-|-$)/g, '');
        }
        
        const anchor = document.createElement('a');
        anchor.className = 'heading-anchor';
        anchor.href = `#${heading.id}`;
        anchor.innerHTML = '<i class="fas fa-link"></i>';
        anchor.title = 'Link to this section';
        heading.appendChild(anchor);
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            if (searchInput) {
                searchInput.focus();
            }
        }
    });
    
    // Add table of contents for long sections
    const longSections = Array.from(sections).filter(section => {
        const headings = section.querySelectorAll('h3, h4');
        return headings.length > 3;
    });
    
    longSections.forEach(section => {
        const headings = section.querySelectorAll('h3, h4');
        if (headings.length > 0) {
            const toc = document.createElement('div');
            toc.className = 'section-toc';
            toc.innerHTML = '<h4>In this section:</h4><ul></ul>';
            const ul = toc.querySelector('ul');
            
            headings.forEach(heading => {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = `#${heading.id}`;
                a.textContent = heading.textContent.replace('', '').trim();
                a.className = heading.tagName === 'H4' ? 'toc-sub' : '';
                li.appendChild(a);
                ul.appendChild(li);
            });
            
            const firstP = section.querySelector('p');
            if (firstP) {
                firstP.parentNode.insertBefore(toc, firstP.nextSibling);
            }
        }
    });
});
