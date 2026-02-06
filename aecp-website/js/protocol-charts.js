// Protocol Page Charts

document.addEventListener('DOMContentLoaded', function() {
    initCalibrationChart();
    initQualityChart();
});

// Calibration Process Chart
function initCalibrationChart() {
    const ctx = document.getElementById('calibrationChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Training (Round-trip)', 'Validation (Round-trip)', 'Test (Unseen)'],
            datasets: [{
                label: 'Semantic Similarity (%)',
                data: [97.35, 97.34, 97.35], // Training round-trip: 97.35%, Validation: 97.34%, Unseen vocab: 97.35%
                borderColor: 'rgba(37, 99, 235, 1)',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                tension: 0.4,
                fill: true,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBackgroundColor: 'rgba(99, 102, 241, 1)',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y + '% similarity';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 90,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Semantic Similarity (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Dataset Split'
                    }
                }
            }
        }
    });
}

// Quality Monitoring Chart
function initQualityChart() {
    const ctx = document.getElementById('qualityChart');
    if (!ctx) return;
    
    // Simulated quality over time
    const days = Array.from({length: 30}, (_, i) => `Day ${i + 1}`);
    const quality = Array.from({length: 30}, () => 97 + (Math.random() - 0.5) * 2);
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: days.filter((_, i) => i % 5 === 0), // Show every 5th day
            datasets: [{
                label: 'Transfer Quality (%)',
                data: quality.filter((_, i) => i % 5 === 0),
                borderColor: 'rgba(16, 185, 129, 1)',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.4,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6
            }, {
                label: 'Quality Threshold',
                data: Array(6).fill(75),
                borderColor: 'rgba(239, 68, 68, 1)',
                borderDash: [5, 5],
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.datasetIndex === 0) {
                                return 'Quality: ' + context.parsed.y.toFixed(2) + '%';
                            }
                            return 'Threshold: ' + context.parsed.y + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 70,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Quality (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Time'
                    }
                }
            }
        }
    });
}
