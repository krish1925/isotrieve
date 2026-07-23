// Transfer Visualization for Home Page

document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('transferVisualization');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = 400;
    
    // Animation variables
    let animationFrame = 0;
    
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = 80;
        
        // Draw source agent (left)
        const sourceX = centerX - 200;
        const sourceY = centerY;
        
        // Source circle
        ctx.beginPath();
        ctx.arc(sourceX, sourceY, radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(37, 99, 235, 0.2)';
        ctx.fill();
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 3;
        ctx.stroke();
        
        // Source label
        ctx.fillStyle = '#1e293b';
        ctx.font = 'bold 16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Agent A', sourceX, sourceY - radius - 20);
        ctx.font = '14px sans-serif';
        ctx.fillText('384d', sourceX, sourceY + radius + 20);
        
        // Draw target agent (right)
        const targetX = centerX + 200;
        const targetY = centerY;
        
        // Target circle
        ctx.beginPath();
        ctx.arc(targetX, targetY, radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
        ctx.fill();
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 3;
        ctx.stroke();
        
        // Target label
        ctx.fillStyle = '#1e293b';
        ctx.font = 'bold 16px sans-serif';
        ctx.fillText('Agent B', targetX, targetY - radius - 20);
        ctx.font = '14px sans-serif';
        ctx.fillText('768d', targetX, targetY + radius + 20);
        
        // Draw transfer arrow
        const arrowStartX = sourceX + radius;
        const arrowEndX = targetX - radius;
        const arrowY = centerY;
        
        // Animated particles
        const particleCount = 5;
        for (let i = 0; i < particleCount; i++) {
            const progress = ((animationFrame + i * 20) % 120) / 120;
            const particleX = arrowStartX + (arrowEndX - arrowStartX) * progress;
            const particleY = arrowY + Math.sin(progress * Math.PI * 2) * 10;
            
            ctx.beginPath();
            ctx.arc(particleX, particleY, 4, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(37, 99, 235, ${1 - progress * 0.5})`;
            ctx.fill();
        }
        
        // Arrow line
        ctx.beginPath();
        ctx.moveTo(arrowStartX, arrowY);
        ctx.lineTo(arrowEndX, arrowY);
        ctx.strokeStyle = '#2563eb';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Arrow head
        const arrowSize = 10;
        ctx.beginPath();
        ctx.moveTo(arrowEndX, arrowY);
        ctx.lineTo(arrowEndX - arrowSize, arrowY - arrowSize / 2);
        ctx.lineTo(arrowEndX - arrowSize, arrowY + arrowSize / 2);
        ctx.closePath();
        ctx.fillStyle = '#2563eb';
        ctx.fill();
        
        // Transfer label
        ctx.fillStyle = '#2563eb';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText('Isotrieve Transfer', centerX, arrowY - 30);
        ctx.font = '12px sans-serif';
        ctx.fillStyle = '#64748b';
        ctx.fillText('97% fidelity • <1ms', centerX, arrowY - 15);
        
        animationFrame++;
        requestAnimationFrame(draw);
    }
    
    draw();
    
    // Handle resize
    window.addEventListener('resize', () => {
        canvas.width = canvas.offsetWidth;
        canvas.height = 400;
    });
});
