// Main JavaScript for AECP Website

document.addEventListener('DOMContentLoaded', function() {
    // Code tabs (main code examples)
    const codeTabs = document.querySelectorAll('.code-tab');
    const codeBlocks = document.querySelectorAll('.code-block');
    
    codeTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const lang = tab.dataset.lang;
            
            // Remove active class from all tabs and blocks in same container
            const container = tab.closest('.code-example');
            if (container) {
                container.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
                container.querySelectorAll('.code-block').forEach(b => b.classList.remove('active'));
            }
            
            // Add active class to clicked tab and corresponding block
            tab.classList.add('active');
            const targetBlock = container ? container.querySelector(`.code-block[data-lang="${lang}"]`) : document.querySelector(`.code-block[data-lang="${lang}"]`);
            if (targetBlock) {
                targetBlock.classList.add('active');
            }
        });
    });
    
    // Sidebar code language tabs
    const sidebarLangTabs = document.querySelectorAll('.code-lang-tab');
    sidebarLangTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const lang = tab.dataset.lang;
            const sidebar = tab.closest('.code-sidebar');
            
            if (sidebar) {
                // Remove active class from all tabs and blocks in this sidebar
                sidebar.querySelectorAll('.code-lang-tab').forEach(t => t.classList.remove('active'));
                sidebar.querySelectorAll('.code-sidebar-block').forEach(b => b.classList.remove('active'));
                
                // Add active class to clicked tab and corresponding block
                tab.classList.add('active');
                const targetBlock = sidebar.querySelector(`.code-sidebar-block[data-lang="${lang}"]`);
                if (targetBlock) {
                    targetBlock.classList.add('active');
                }
            }
        });
    });
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
});
