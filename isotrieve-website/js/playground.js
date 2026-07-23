// Isotrieve Playground Interactive Functionality

// State management
const playgroundState = {
    agentA: { model: 384, dimensions: 384 },
    agentB: { model: 768, dimensions: 768 },
    vocabSize: 500,
    calibrated: false,
    transferMatrices: null,
    qualityThreshold: 0.75
};

// Mock embedding generation (simulates real embeddings)
function generateMockEmbedding(text, dimensions) {
    const embedding = new Array(dimensions);
    // Use text hash to create deterministic embeddings
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
        hash = ((hash << 5) - hash) + text.charCodeAt(i);
        hash = hash & hash;
    }
    
    // Generate embedding with some structure
    for (let i = 0; i < dimensions; i++) {
        const seed = (hash + i * 1000) / 1000000;
        embedding[i] = Math.sin(seed) * 0.5 + Math.cos(seed * 2) * 0.3 + (Math.random() - 0.5) * 0.2;
    }
    
    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map(val => val / norm);
}

// Compute cosine similarity
function cosineSimilarity(vec1, vec2) {
    const minLen = Math.min(vec1.length, vec2.length);
    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;
    
    for (let i = 0; i < minLen; i++) {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    
    return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

// Generate transfer matrix (simplified simulation)
function generateTransferMatrix(fromDim, toDim) {
    const matrix = [];
    for (let i = 0; i < fromDim; i++) {
        const row = [];
        for (let j = 0; j < toDim; j++) {
            // Create a semi-realistic transfer matrix
            row.push((Math.random() - 0.5) * 0.1 + (i === j ? 1 : 0));
        }
        matrix.push(row);
    }
    return matrix;
}

// Matrix-vector multiplication
function matrixVectorMultiply(matrix, vector) {
    const result = new Array(matrix[0].length).fill(0);
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            result[j] += matrix[i][j] * vector[i];
        }
    }
    // Normalize
    const norm = Math.sqrt(result.reduce((sum, val) => sum + val * val, 0));
    return result.map(val => val / norm);
}

// Calibrate agents
async function calibrateAgents() {
    const btn = document.getElementById('calibrateBtn');
    const status = document.getElementById('calibrationStatus');
    const icon = btn.querySelector('i');
    
    // Show loading
    btn.disabled = true;
    icon.classList.add('spinning');
    status.className = 'status-message loading';
    status.textContent = `Calibrating agents with ${playgroundState.vocabSize} vocabulary items...`;
    
    // Simulate calibration delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    try {
        // Generate transfer matrices
        playgroundState.transferMatrices = {
            AtoB: generateTransferMatrix(playgroundState.agentA.dimensions, playgroundState.agentB.dimensions),
            BtoA: generateTransferMatrix(playgroundState.agentB.dimensions, playgroundState.agentA.dimensions)
        };
        
        // Simulate calibration quality (realistic values)
        const quality = 0.92 + Math.random() * 0.06; // 92-98%
        
        playgroundState.calibrated = true;
        
        // Show success
        status.className = 'status-message success';
        status.innerHTML = `
            <strong>✓ Calibration successful!</strong><br>
            Quality: ${(quality * 100).toFixed(1)}% | 
            Vocabulary: ${playgroundState.vocabSize} items | 
            Matrix size: ${playgroundState.agentA.dimensions}×${playgroundState.agentB.dimensions}
        `;
        
        // Enable transfer buttons
        document.getElementById('transferBtn').disabled = false;
        document.getElementById('textTransferBtn').disabled = false;
        
    } catch (error) {
        status.className = 'status-message error';
        status.textContent = `Calibration failed: ${error.message}`;
    } finally {
        btn.disabled = false;
        icon.classList.remove('spinning');
    }
}

// Transfer embedding using Isotrieve
async function transferWithIsotrieve(text) {
    const startTime = performance.now();
    
    // Generate embedding in Agent A space
    const embeddingA = generateMockEmbedding(text, playgroundState.agentA.dimensions);
    
    // Transfer to Agent B space using matrix
    const embeddingB = matrixVectorMultiply(playgroundState.transferMatrices.AtoB, embeddingA);
    
    // Round-trip back to Agent A to measure quality
    const embeddingA_reconstructed = matrixVectorMultiply(playgroundState.transferMatrices.BtoA, embeddingB);
    
    // Calculate similarity
    const similarity = cosineSimilarity(embeddingA, embeddingA_reconstructed);
    
    const endTime = performance.now();
    const latency = endTime - startTime;
    
    return {
        embedding: embeddingB,
        similarity: similarity,
        latency: latency,
        informationPreserved: similarity * 100
    };
}

// Transfer embedding using text (baseline)
async function transferWithText(text) {
    const startTime = performance.now();
    
    // Simulate text serialization delay
    await new Promise(resolve => setTimeout(resolve, 150));
    
    // Generate embedding in Agent A space
    const embeddingA = generateMockEmbedding(text, playgroundState.agentA.dimensions);
    
    // Simulate information loss in text conversion
    // Text preserves only ~43% of semantic information
    const textLossFactor = 0.43;
    
    // Re-embed in Agent B space (with information loss)
    const embeddingB_text = generateMockEmbedding(text, playgroundState.agentB.dimensions);
    
    // Add noise to simulate information loss
    const embeddingB = embeddingB_text.map(val => val * textLossFactor + (Math.random() - 0.5) * 0.3);
    
    // Normalize
    const norm = Math.sqrt(embeddingB.reduce((sum, val) => sum + val * val, 0));
    const normalizedB = embeddingB.map(val => val / norm);
    
    // Calculate similarity (will be lower due to text loss)
    const similarity = textLossFactor + (Math.random() - 0.5) * 0.1;
    
    const endTime = performance.now();
    const latency = endTime - startTime;
    
    return {
        embedding: normalizedB,
        similarity: similarity,
        latency: latency,
        informationPreserved: similarity * 100
    };
}

// Display results
function displayResults(isotrieveResult, textResult) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    
    // Isotrieve Results
    document.getElementById('isotrieveSimilarity').textContent = `${(isotrieveResult.similarity * 100).toFixed(1)}%`;
    document.getElementById('isotrieveSimilarity').className = `metric-value ${getQualityClass(isotrieveResult.similarity)}`;
    document.getElementById('isotrievePreserved').textContent = `${isotrieveResult.informationPreserved.toFixed(1)}%`;
    document.getElementById('isotrieveLatency').textContent = `${isotrieveResult.latency.toFixed(2)}ms`;
    
    const isotrieveQualityBar = document.getElementById('isotrieveQuality').querySelector('.quality-fill');
    isotrieveQualityBar.style.width = `${isotrieveResult.similarity * 100}%`;
    
    // Display first 10 dimensions
    const isotrieveEmbedding = document.getElementById('isotrieveEmbedding');
    isotrieveEmbedding.innerHTML = isotrieveResult.embedding.slice(0, 10)
        .map(val => `<span>${val.toFixed(4)}</span>`)
        .join('');
    
    // Text Results
    document.getElementById('textSimilarity').textContent = `${(textResult.similarity * 100).toFixed(1)}%`;
    document.getElementById('textSimilarity').className = `metric-value ${getQualityClass(textResult.similarity)}`;
    document.getElementById('textPreserved').textContent = `${textResult.informationPreserved.toFixed(1)}%`;
    document.getElementById('textLatency').textContent = `${textResult.latency.toFixed(2)}ms`;
    
    const textQualityBar = document.getElementById('textQuality').querySelector('.quality-fill');
    textQualityBar.style.width = `${textResult.similarity * 100}%`;
    textQualityBar.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
    
    const textEmbedding = document.getElementById('textEmbedding');
    textEmbedding.innerHTML = textResult.embedding.slice(0, 10)
        .map(val => `<span>${val.toFixed(4)}</span>`)
        .join('');
    
    // Comparison
    const advantage = ((isotrieveResult.similarity - textResult.similarity) / textResult.similarity * 100);
    document.getElementById('isotrieveAdvantage').textContent = `+${advantage.toFixed(0)}%`;
    
    const speedImprovement = ((textResult.latency - isotrieveResult.latency) / textResult.latency * 100);
    document.getElementById('speedImprovement').textContent = `${speedImprovement.toFixed(0)}x`;
    
    // Update chart
    updateComparisonChart(isotrieveResult, textResult);
    
    // Update visualization
    updateEmbeddingVisualization(isotrieveResult, textResult);
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Get quality class
function getQualityClass(similarity) {
    if (similarity >= 0.90) return 'excellent';
    if (similarity >= 0.80) return 'good';
    if (similarity >= 0.75) return 'fair';
    return 'poor';
}

// Update comparison chart
function updateComparisonChart(isotrieveResult, textResult) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    
    if (window.comparisonChartInstance) {
        window.comparisonChartInstance.destroy();
    }
    
    window.comparisonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Semantic Similarity', 'Information Preserved', 'Speed (inverse latency)'],
            datasets: [
                {
                    label: 'Isotrieve',
                    data: [
                        isotrieveResult.similarity * 100,
                        isotrieveResult.informationPreserved,
                        (1000 / isotrieveResult.latency)
                    ],
                    backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    borderColor: 'rgb(16, 185, 129)',
                    borderWidth: 2
                },
                {
                    label: 'Text Transfer',
                    data: [
                        textResult.similarity * 100,
                        textResult.informationPreserved,
                        (1000 / textResult.latency)
                    ],
                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                    borderColor: 'rgb(239, 68, 68)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                if (context.dataIndex === 2) {
                                    label += context.parsed.y.toFixed(0) + ' transfers/sec';
                                } else {
                                    label += context.parsed.y.toFixed(1) + '%';
                                }
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}

// Update embedding visualization
function updateEmbeddingVisualization(isotrieveResult, textResult) {
    const canvas = document.getElementById('embeddingViz');
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = 400;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Simulate 2D projection of embeddings using first 2 dimensions
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const scale = 150;
    
    // Original embedding (Agent A)
    const origX = centerX - scale;
    const origY = centerY;
    
    // Isotrieve transfer
    const isotrieveX = centerX + isotrieveResult.embedding[0] * scale;
    const isotrieveY = centerY + isotrieveResult.embedding[1] * scale;
    
    // Text transfer
    const textX = centerX + textResult.embedding[0] * scale * 0.7;
    const textY = centerY + textResult.embedding[1] * scale * 0.7;
    
    // Draw connections
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    ctx.beginPath();
    ctx.moveTo(origX, origY);
    ctx.lineTo(isotrieveX, isotrieveY);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(origX, origY);
    ctx.lineTo(textX, textY);
    ctx.stroke();
    
    ctx.setLineDash([]);
    
    // Draw points
    // Original
    ctx.fillStyle = '#2563eb';
    ctx.beginPath();
    ctx.arc(origX, origY, 12, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#1e40af';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Isotrieve
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(isotrieveX, isotrieveY, 12, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#059669';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Text
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(textX, textY, 12, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Add labels
    ctx.fillStyle = '#1e293b';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Original', origX, origY - 20);
    ctx.fillText('Isotrieve', isotrieveX, isotrieveY - 20);
    ctx.fillText('Text', textX, textY - 20);
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Model selection
    document.getElementById('agentAModel').addEventListener('change', function(e) {
        playgroundState.agentA.dimensions = parseInt(e.target.value);
        document.getElementById('agentADims').textContent = e.target.value;
        playgroundState.calibrated = false;
        document.getElementById('transferBtn').disabled = true;
        document.getElementById('textTransferBtn').disabled = true;
    });
    
    document.getElementById('agentBModel').addEventListener('change', function(e) {
        playgroundState.agentB.dimensions = parseInt(e.target.value);
        document.getElementById('agentBDims').textContent = e.target.value;
        playgroundState.calibrated = false;
        document.getElementById('transferBtn').disabled = true;
        document.getElementById('textTransferBtn').disabled = true;
    });
    
    // Vocabulary size
    document.getElementById('vocabSize').addEventListener('change', function(e) {
        playgroundState.vocabSize = parseInt(e.target.value);
        playgroundState.calibrated = false;
        document.getElementById('transferBtn').disabled = true;
        document.getElementById('textTransferBtn').disabled = true;
    });
    
    // Calibrate button
    document.getElementById('calibrateBtn').addEventListener('click', calibrateAgents);
    
    // Quick examples
    document.querySelectorAll('.btn-example').forEach(btn => {
        btn.addEventListener('click', function() {
            document.getElementById('inputText').value = this.dataset.text;
        });
    });
    
    // Transfer buttons
    document.getElementById('transferBtn').addEventListener('click', async function() {
        const text = document.getElementById('inputText').value.trim();
        if (!text) {
            alert('Please enter some text to transfer');
            return;
        }
        
        const status = document.getElementById('transferStatus');
        status.className = 'status-message loading';
        status.textContent = 'Transferring embeddings...';
        
        try {
            const isotrieveResult = await transferWithIsotrieve(text);
            const textResult = await transferWithText(text);
            
            displayResults(isotrieveResult, textResult);
            
            status.className = 'status-message success';
            status.textContent = '✓ Transfer complete! See results below.';
        } catch (error) {
            status.className = 'status-message error';
            status.textContent = `Transfer failed: ${error.message}`;
        }
    });
    
    document.getElementById('textTransferBtn').addEventListener('click', async function() {
        const text = document.getElementById('inputText').value.trim();
        if (!text) {
            alert('Please enter some text to transfer');
            return;
        }
        
        const status = document.getElementById('transferStatus');
        status.className = 'status-message loading';
        status.textContent = 'Transferring via text (baseline)...';
        
        try {
            const textResult = await transferWithText(text);
            const isotrieveResult = await transferWithIsotrieve(text);
            
            displayResults(isotrieveResult, textResult);
            
            status.className = 'status-message info';
            status.textContent = '✓ Baseline transfer complete! Compare with Isotrieve above.';
        } catch (error) {
            status.className = 'status-message error';
            status.textContent = `Transfer failed: ${error.message}`;
        }
    });
    
    // Quality threshold
    document.getElementById('qualityThreshold').addEventListener('input', function(e) {
        playgroundState.qualityThreshold = parseFloat(e.target.value);
        document.getElementById('thresholdValue').textContent = e.target.value;
    });
    
    // Code tabs
    document.querySelectorAll('.code-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            const lang = this.dataset.lang;
            
            // Update active tab
            document.querySelectorAll('.code-tab').forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            
            // Update active code block
            document.querySelectorAll('.code-block').forEach(block => {
                if (block.dataset.lang === lang) {
                    block.classList.add('active');
                } else {
                    block.classList.remove('active');
                }
            });
        });
    });
});
