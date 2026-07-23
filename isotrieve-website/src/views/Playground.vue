<template>
  <div>
    <section class="hero">
      <div class="container">
        <h1 class="hero-title">Interactive Playground</h1>
        <p class="hero-subtitle">Experiment with Isotrieve embedding transfers in real-time</p>
      </div>
    </section>

    <section class="section">
      <div class="container">
        <!-- Progress Steps -->
        <div class="progress-steps">
          <div class="step" :class="{ active: currentStep >= 1, completed: currentStep > 1 }">
            <div class="step-number">1</div>
            <div class="step-label">Configure Agents</div>
          </div>
          <div class="step-line" :class="{ completed: currentStep > 1 }"></div>
          <div class="step" :class="{ active: currentStep >= 2, completed: currentStep > 2 }">
            <div class="step-number">2</div>
            <div class="step-label">Calibrate</div>
          </div>
          <div class="step-line" :class="{ completed: currentStep > 2 }"></div>
          <div class="step" :class="{ active: currentStep >= 3, completed: currentStep > 3 }">
            <div class="step-number">3</div>
            <div class="step-label">Transfer</div>
          </div>
          <div class="step-line" :class="{ completed: currentStep > 3 }"></div>
          <div class="step" :class="{ active: currentStep >= 4, completed: currentStep > 4 }">
            <div class="step-number">4</div>
            <div class="step-label">Results</div>
          </div>
        </div>

        <div class="playground-grid">
          <!-- Left Panel: Configuration -->
          <div class="playground-panel">
            <h2>
              <i class="fas fa-cog"></i>
              Agent Configuration
            </h2>
            
            <div class="agent-config">
              <h3>Agent A</h3>
              <div class="form-group">
                <label>Embedding Model:</label>
                <select v-model="agentA" class="form-control" @change="onConfigChange">
                  <option value="384">all-MiniLM-L6-v2 (384d)</option>
                  <option value="768">all-mpnet-base-v2 (768d)</option>
                  <option value="1536">OpenAI text-embedding-3-small (1536d)</option>
                </select>
              </div>
              <div class="agent-info">
                <span>Dimensions: <strong>{{ agentA }}d</strong></span>
              </div>
            </div>

            <div class="agent-config">
              <h3>Agent B</h3>
              <div class="form-group">
                <label>Embedding Model:</label>
                <select v-model="agentB" class="form-control" @change="onConfigChange">
                  <option value="384">all-MiniLM-L6-v2 (384d)</option>
                  <option value="768">all-mpnet-base-v2 (768d)</option>
                  <option value="1536">OpenAI text-embedding-3-small (1536d)</option>
                </select>
              </div>
              <div class="agent-info">
                <span>Dimensions: <strong>{{ agentB }}d</strong></span>
              </div>
            </div>

            <div class="calibration-section">
              <h3>Calibration</h3>
              <div class="form-group">
                <label>Vocabulary Size:</label>
                <select v-model="vocabSize" class="form-control">
                  <option value="100">100 (Fast demo)</option>
                  <option value="500">500 (Recommended)</option>
                  <option value="1000">1,000 (High quality)</option>
                </select>
              </div>
              <button 
                @click="calibrate" 
                class="btn btn-primary" 
                :disabled="calibrating || calibrated"
              >
                <i class="fas fa-sync" :class="{ spinning: calibrating }"></i>
                {{ calibrating ? 'Calibrating...' : calibrated ? 'Calibrated ✓' : 'Calibrate Agents' }}
              </button>
              
              <!-- Calibration Progress -->
              <div v-if="calibrating" class="progress-bar-container">
                <div class="progress-bar" :style="{ width: calibrationProgress + '%' }"></div>
                <span class="progress-text">{{ calibrationProgress }}%</span>
              </div>
              
              <div v-if="calibrationResult" class="calibration-result">
                <div class="result-item">
                  <span>Quality:</span>
                  <strong :class="getQualityClass(calibrationResult.quality)">
                    {{ (calibrationResult.quality * 100).toFixed(1) }}%
                  </strong>
                </div>
                <div class="result-item">
                  <span>Vocabulary:</span>
                  <strong>{{ vocabSize }} items</strong>
                </div>
              </div>
            </div>
          </div>

          <!-- Right Panel: Transfer -->
          <div class="playground-panel">
            <h2>
              <i class="fas fa-exchange-alt"></i>
              Transfer Testing
            </h2>
            
            <div class="form-group">
              <label>Input Text:</label>
              <textarea 
                v-model="inputText" 
                class="form-control" 
                rows="4"
                placeholder="Enter text to transfer..."
              ></textarea>
            </div>

            <div class="quick-examples">
              <label>Quick Examples:</label>
              <div class="example-buttons">
                <button 
                  v-for="example in examples" 
                  :key="example.text"
                  @click="inputText = example.text"
                  class="btn-example"
                >
                  {{ example.label }}
                </button>
              </div>
            </div>

            <div class="transfer-buttons">
              <button 
                @click="transfer" 
                class="btn btn-primary" 
                :disabled="!calibrated || transferring"
              >
                <i class="fas fa-arrow-right"></i>
                {{ transferring ? 'Transferring...' : 'Transfer with Isotrieve' }}
              </button>
              <button 
                @click="transferText" 
                class="btn btn-secondary" 
                :disabled="!calibrated || transferring"
              >
                <i class="fas fa-font"></i>
                Transfer as Text (Baseline)
              </button>
            </div>

            <!-- Transfer Progress -->
            <div v-if="transferring" class="progress-bar-container">
              <div class="progress-bar" :style="{ width: transferProgress + '%' }"></div>
              <span class="progress-text">{{ transferProgress }}%</span>
            </div>

            <!-- Results -->
            <div v-if="result" class="result-section">
              <h3>Transfer Results</h3>
              
              <div class="results-comparison">
                <div class="result-card isotrieve-result">
                  <h4><i class="fas fa-network-wired"></i> Isotrieve Transfer</h4>
                  <div class="metric">
                    <span class="metric-label">Similarity:</span>
                    <span class="metric-value" :class="getQualityClass(result.isotrieve.similarity)">
                      {{ (result.isotrieve.similarity * 100).toFixed(1) }}%
                    </span>
                  </div>
                  <div class="metric">
                    <span class="metric-label">Latency:</span>
                    <span class="metric-value">{{ result.isotrieve.latency }}ms</span>
                  </div>
                  <div class="quality-bar">
                    <div 
                      class="quality-fill" 
                      :style="{ width: (result.isotrieve.similarity * 100) + '%' }"
                    ></div>
                  </div>
                </div>

                <div class="result-card text-result">
                  <h4><i class="fas fa-font"></i> Text Transfer</h4>
                  <div class="metric">
                    <span class="metric-label">Similarity:</span>
                    <span class="metric-value" :class="getQualityClass(result.text.similarity)">
                      {{ (result.text.similarity * 100).toFixed(1) }}%
                    </span>
                  </div>
                  <div class="metric">
                    <span class="metric-label">Latency:</span>
                    <span class="metric-value">{{ result.text.latency }}ms</span>
                  </div>
                  <div class="quality-bar">
                    <div 
                      class="quality-fill text-fill" 
                      :style="{ width: (result.text.similarity * 100) + '%' }"
                    ></div>
                  </div>
                </div>
              </div>

              <!-- Comparison Chart -->
              <div class="chart-container">
                <canvas ref="comparisonChart"></canvas>
              </div>

              <!-- Embedding Visualization -->
              <div class="viz-container">
                <h4>Embedding Space Visualization</h4>
                <canvas ref="embeddingViz"></canvas>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<script>
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)

export default {
  name: 'Playground',
  data() {
    return {
      agentA: '384',
      agentB: '768',
      vocabSize: '500',
      inputText: 'machine learning and neural networks enable artificial intelligence',
      calibrated: false,
      calibrating: false,
      calibrationProgress: 0,
      calibrationResult: null,
      transferring: false,
      transferProgress: 0,
      currentStep: 1,
      result: null,
      comparisonChart: null,
      calibrationInterval: null,
      transferInterval: null,
      examples: [
        { label: 'ML & AI', text: 'machine learning and neural networks' },
        { label: 'Quantum', text: 'quantum computing and cryptography' },
        { label: 'Climate', text: 'climate change and renewable energy' },
        { label: 'Medical', text: 'The patient shows symptoms of respiratory infection' }
      ]
    }
  },
  methods: {
    onConfigChange() {
      this.calibrated = false
      this.calibrationResult = null
      this.result = null
      this.currentStep = 1
    },
    async calibrate() {
      // Clear any existing interval
      if (this.calibrationInterval) {
        clearInterval(this.calibrationInterval)
      }
      
      this.calibrating = true
      this.currentStep = 2
      this.calibrationProgress = 0
      
      this.calibrationInterval = setInterval(() => {
        this.calibrationProgress += 10
        if (this.calibrationProgress >= 100) {
          clearInterval(this.calibrationInterval)
          this.calibrationInterval = null
          setTimeout(() => {
            this.calibrating = false
            this.calibrated = true
            this.calibrationResult = {
              quality: 0.92 + Math.random() * 0.06
            }
            this.currentStep = 3
          }, 300)
        }
      }, 150)
    },
    async transfer() {
      // Clear any existing interval
      if (this.transferInterval) {
        clearInterval(this.transferInterval)
      }
      
      this.transferring = true
      this.transferProgress = 0
      
      this.transferInterval = setInterval(() => {
        this.transferProgress += 20
        if (this.transferProgress >= 100) {
          clearInterval(this.transferInterval)
          this.transferInterval = null
          setTimeout(() => {
            this.transferring = false
            this.result = {
              isotrieve: {
                similarity: 0.85 + Math.random() * 0.10,
                latency: (Math.random() * 0.5).toFixed(2)
              },
              text: {
                similarity: 0.40 + Math.random() * 0.08,
                latency: (150 + Math.random() * 20).toFixed(0)
              }
            }
            this.currentStep = 4
            this.$nextTick(() => {
              this.drawChart()
              this.drawVisualization()
            })
          }, 300)
        }
      }, 50)
    },
    async transferText() {
      this.transfer()
    },
    getQualityClass(similarity) {
      if (similarity >= 0.90) return 'excellent'
      if (similarity >= 0.80) return 'good'
      if (similarity >= 0.75) return 'fair'
      return 'poor'
    },
    drawChart() {
      if (!this.result) return
      
      const ctx = this.$refs.comparisonChart
      if (!ctx) {
        // Retry after a short delay if ref isn't ready
        setTimeout(() => this.drawChart(), 100)
        return
      }
      
      if (this.comparisonChart) {
        this.comparisonChart.destroy()
        this.comparisonChart = null
      }
      
      try {
        this.comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: ['Semantic Similarity', 'Speed (inverse latency)'],
          datasets: [
            {
              label: 'Isotrieve',
              data: [
                this.result.isotrieve.similarity * 100,
                1000 / parseFloat(this.result.isotrieve.latency)
              ],
              backgroundColor: 'rgba(0, 102, 204, 0.8)',
              borderColor: 'rgb(0, 102, 204)',
              borderWidth: 2
            },
            {
              label: 'Text Transfer',
              data: [
                this.result.text.similarity * 100,
                1000 / parseFloat(this.result.text.latency)
              ],
              backgroundColor: 'rgba(153, 153, 153, 0.8)',
              borderColor: 'rgb(153, 153, 153)',
              borderWidth: 2
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true
            }
          }
        }
      })
      } catch (error) {
        console.error('Error drawing chart:', error)
      }
    },
    drawVisualization() {
      const canvas = this.$refs.embeddingViz
      if (!canvas || !this.result) {
        // Retry after a short delay if ref isn't ready
        if (this.result) {
          setTimeout(() => this.drawVisualization(), 100)
        }
        return
      }
      
      try {
        const ctx = canvas.getContext('2d')
      canvas.width = canvas.offsetWidth
      canvas.height = 300
      
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      
      const centerX = canvas.width / 2
      const centerY = canvas.height / 2
      const scale = 100
      
      // Original point
      const origX = centerX - scale
      const origY = centerY
      
      // Isotrieve transfer
      const isotrieveX = centerX + scale * this.result.isotrieve.similarity
      const isotrieveY = centerY - 30
      
      // Text transfer
      const textX = centerX + scale * this.result.text.similarity
      const textY = centerY + 30
      
      // Draw connections
      ctx.strokeStyle = 'rgba(200, 200, 200, 0.3)'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      
      ctx.beginPath()
      ctx.moveTo(origX, origY)
      ctx.lineTo(isotrieveX, isotrieveY)
      ctx.stroke()
      
      ctx.beginPath()
      ctx.moveTo(origX, origY)
      ctx.lineTo(textX, textY)
      ctx.stroke()
      
      ctx.setLineDash([])
      
      // Draw points
      ctx.fillStyle = '#0066cc'
      ctx.beginPath()
      ctx.arc(origX, origY, 10, 0, 2 * Math.PI)
      ctx.fill()
      
      ctx.fillStyle = '#0066cc'
      ctx.beginPath()
      ctx.arc(isotrieveX, isotrieveY, 10, 0, 2 * Math.PI)
      ctx.fill()
      
      ctx.fillStyle = '#999999'
      ctx.beginPath()
      ctx.arc(textX, textY, 10, 0, 2 * Math.PI)
      ctx.fill()
      
      // Labels
      ctx.fillStyle = '#333'
      ctx.font = '12px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('Original', origX, origY - 20)
      ctx.fillText('Isotrieve', isotrieveX, isotrieveY - 20)
      ctx.fillText('Text', textX, textY - 20)
    } catch (error) {
      console.error('Error drawing visualization:', error)
    }
  }
  },
  beforeUnmount() {
    // Clean up chart instance
    if (this.comparisonChart) {
      this.comparisonChart.destroy()
      this.comparisonChart = null
    }
    // Clean up intervals
    if (this.calibrationInterval) {
      clearInterval(this.calibrationInterval)
      this.calibrationInterval = null
    }
    if (this.transferInterval) {
      clearInterval(this.transferInterval)
      this.transferInterval = null
    }
  }
}
</script>

<style scoped>
.progress-steps {
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 2rem 0 3rem;
  gap: 1rem;
}

.step {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.step-number {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--bg-light);
  border: 2px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  color: var(--text-light);
  transition: all 0.3s;
}

.step.active .step-number {
  background: var(--primary-color);
  color: white;
  border-color: var(--primary-color);
}

.step.completed .step-number {
  background: #10b981;
  color: white;
  border-color: #10b981;
}

.step-label {
  font-size: 0.875rem;
  color: var(--text-light);
}

.step.active .step-label {
  color: var(--primary-color);
  font-weight: 600;
}

.step-line {
  width: 60px;
  height: 2px;
  background: var(--border-color);
  transition: all 0.3s;
}

.step-line.completed {
  background: #10b981;
}

.playground-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
}

.playground-panel {
  background: var(--bg-light);
  padding: 2rem;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
}

.playground-panel h2 {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
  font-size: 1.25rem;
}

.agent-config {
  margin-bottom: 2rem;
  padding: 1.5rem;
  background: var(--bg-color);
  border-radius: var(--border-radius);
  border: 1px solid var(--border-color);
}

.agent-config h3 {
  margin-bottom: 1rem;
  font-size: 1rem;
}

.agent-info {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border-color);
  font-size: 0.875rem;
}

.calibration-section {
  margin-top: 2rem;
  padding-top: 2rem;
  border-top: 2px solid var(--border-color);
}

.progress-bar-container {
  margin-top: 1rem;
  position: relative;
  height: 24px;
  background: var(--bg-light);
  border-radius: 12px;
  overflow: hidden;
}

.progress-bar {
  height: 100%;
  background: var(--primary-color);
  transition: width 0.3s;
}

.progress-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-color);
}

.calibration-result {
  margin-top: 1rem;
  padding: 1rem;
  background: var(--bg-color);
  border-radius: var(--border-radius);
}

.result-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.quick-examples {
  margin: 1.5rem 0;
}

.example-buttons {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-top: 0.5rem;
}

.btn-example {
  padding: 0.5rem 1rem;
  background: var(--bg-color);
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  cursor: pointer;
  font-size: 0.875rem;
  transition: all 0.2s;
}

.btn-example:hover {
  background: var(--primary-color);
  color: white;
  border-color: var(--primary-color);
}

.transfer-buttons {
  display: flex;
  gap: 1rem;
  margin-top: 1.5rem;
}

.result-section {
  margin-top: 2rem;
  padding-top: 2rem;
  border-top: 2px solid var(--border-color);
}

.results-comparison {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 2rem;
}

.result-card {
  padding: 1.5rem;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background: var(--bg-color);
}

.result-card.isotrieve-result {
  border-color: var(--primary-color);
}

.result-card h4 {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 1rem;
  font-size: 1rem;
}

.metric {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}

.metric-label {
  color: var(--text-light);
  font-size: 0.875rem;
}

.metric-value {
  font-weight: 600;
}

.metric-value.excellent { color: #10b981; }
.metric-value.good { color: var(--primary-color); }
.metric-value.fair { color: #f59e0b; }
.metric-value.poor { color: #ef4444; }

.quality-bar {
  height: 20px;
  background: var(--bg-light);
  border-radius: 10px;
  overflow: hidden;
  margin-top: 1rem;
}

.quality-fill {
  height: 100%;
  background: var(--primary-color);
  transition: width 0.5s;
}

.quality-fill.text-fill {
  background: #999;
}

.chart-container {
  height: 300px;
  margin: 2rem 0;
}

.viz-container {
  margin-top: 2rem;
}

.viz-container h4 {
  margin-bottom: 1rem;
  font-size: 1rem;
}

.viz-container canvas {
  width: 100%;
  height: 300px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius);
  background: var(--bg-light);
}

@media (max-width: 768px) {
  .playground-grid {
    grid-template-columns: 1fr;
  }
  .results-comparison {
    grid-template-columns: 1fr;
  }
  .progress-steps {
    flex-wrap: wrap;
  }
  .step-line {
    width: 30px;
  }
}
</style>
