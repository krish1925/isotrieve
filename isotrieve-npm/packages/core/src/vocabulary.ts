/**
 * Default calibration vocabulary for Isotrieve
 * Curated from MTEB benchmark for semantic diversity
 */

export const DEFAULT_VOCABULARY = [
  // Core concepts
  'information', 'knowledge', 'understanding', 'learning', 'reasoning',
  'communication', 'language', 'meaning', 'context', 'semantics',
  
  // Technical terms
  'algorithm', 'data', 'model', 'system', 'network',
  'architecture', 'framework', 'implementation', 'optimization', 'performance',
  
  // Actions
  'analyze', 'compute', 'process', 'transform', 'generate',
  'evaluate', 'measure', 'compare', 'classify', 'predict',
  
  // Abstract concepts
  'similarity', 'distance', 'relationship', 'structure', 'pattern',
  'feature', 'dimension', 'space', 'vector', 'matrix',
  
  // Scientific
  'research', 'experiment', 'hypothesis', 'theory', 'evidence',
  'method', 'analysis', 'result', 'conclusion', 'validation',
  
  // Quality metrics
  'accuracy', 'precision', 'recall', 'quality', 'reliability',
  'consistency', 'robustness', 'efficiency', 'scalability', 'stability',
  
  // Common verbs
  'create', 'build', 'develop', 'design', 'implement',
  'test', 'verify', 'validate', 'optimize', 'improve',
  
  // Common nouns
  'problem', 'solution', 'approach', 'strategy', 'technique',
  'tool', 'resource', 'component', 'module', 'interface',
  
  // Descriptive
  'complex', 'simple', 'efficient', 'effective', 'robust',
  'flexible', 'scalable', 'reliable', 'accurate', 'precise',
  
  // Domain-specific (ML/AI)
  'embedding', 'representation', 'encoding', 'decoding', 'transfer',
  'training', 'inference', 'prediction', 'classification', 'clustering',
  
  // More technical
  'neural', 'transformer', 'attention', 'layer', 'parameter',
  'gradient', 'loss', 'objective', 'constraint', 'regularization',
  
  // Data-related
  'dataset', 'sample', 'instance', 'example', 'observation',
  'feature', 'attribute', 'label', 'target', 'output',
  
  // Mathematical
  'function', 'equation', 'variable', 'constant', 'coefficient',
  'derivative', 'integral', 'probability', 'distribution', 'statistics',
  
  // Phrases for better coverage
  'machine learning', 'artificial intelligence', 'natural language processing',
  'computer vision', 'deep learning', 'neural network', 'data science',
  'semantic similarity', 'vector space', 'dimensionality reduction',
  
  // Common sentences
  'The model performs well on test data.',
  'We need to optimize the algorithm for better performance.',
  'The system achieves high accuracy on the benchmark.',
  'This approach is more efficient than previous methods.',
  'The results demonstrate significant improvement.',
  
  // Technical sentences
  'The embedding space captures semantic relationships between words.',
  'Transfer learning enables knowledge sharing across different tasks.',
  'The attention mechanism focuses on relevant parts of the input.',
  'Gradient descent optimizes the model parameters iteratively.',
  'The loss function measures the difference between predictions and targets.',
  
  // Additional diversity
  'quantum', 'classical', 'theoretical', 'practical', 'empirical',
  'synthetic', 'organic', 'digital', 'analog', 'hybrid',
  'sequential', 'parallel', 'distributed', 'centralized', 'decentralized',
  'supervised', 'unsupervised', 'reinforcement', 'semi-supervised', 'self-supervised',
  
  // More sentences for robustness
  'The architecture consists of multiple layers with different functions.',
  'We evaluate the model using standard benchmark datasets.',
  'The training process converges after several epochs.',
  'Cross-validation helps assess model generalization.',
  'The framework supports various types of neural networks.',
  
  // Edge cases
  'a', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
  'hello world', 'test case', 'example input', 'sample output',
  
  // Numbers and special
  'zero', 'one', 'two', 'three', 'first', 'second', 'third',
  'alpha', 'beta', 'gamma', 'delta', 'epsilon',
];

/**
 * Generate extended vocabulary by combining base terms
 */
export function generateExtendedVocabulary(size: number = 1000): string[] {
  const vocab = [...DEFAULT_VOCABULARY];
  
  const prefixes = ['pre', 'post', 'multi', 'sub', 'inter', 'intra', 'super', 'meta'];
  const suffixes = ['tion', 'ing', 'ed', 'able', 'ness', 'ity', 'ment', 'ive'];
  const adjectives = ['advanced', 'basic', 'novel', 'standard', 'custom', 'generic'];
  const nouns = ['system', 'model', 'method', 'approach', 'technique', 'framework'];
  
  // Generate combinations
  while (vocab.length < size) {
    const type = Math.floor(Math.random() * 4);
    
    if (type === 0 && prefixes.length > 0 && DEFAULT_VOCABULARY.length > 0) {
      // Prefix combinations
      const prefix = prefixes[Math.floor(Math.random() * prefixes.length)];
      const base = DEFAULT_VOCABULARY[Math.floor(Math.random() * DEFAULT_VOCABULARY.length)];
      vocab.push(`${prefix}${base}`);
    } else if (type === 1 && adjectives.length > 0 && nouns.length > 0) {
      // Adjective-noun combinations
      const adj = adjectives[Math.floor(Math.random() * adjectives.length)];
      const noun = nouns[Math.floor(Math.random() * nouns.length)];
      vocab.push(`${adj} ${noun}`);
    } else if (type === 2 && DEFAULT_VOCABULARY.length > 1) {
      // Two-word combinations
      const word1 = DEFAULT_VOCABULARY[Math.floor(Math.random() * DEFAULT_VOCABULARY.length)];
      const word2 = DEFAULT_VOCABULARY[Math.floor(Math.random() * DEFAULT_VOCABULARY.length)];
      if (word1 !== word2) {
        vocab.push(`${word1} ${word2}`);
      }
    } else if (DEFAULT_VOCABULARY.length > 0) {
      // Add from default with slight variation
      const base = DEFAULT_VOCABULARY[Math.floor(Math.random() * DEFAULT_VOCABULARY.length)];
      vocab.push(`${base} example`);
    }
  }
  
  // Remove duplicates and return
  return Array.from(new Set(vocab)).slice(0, size);
}
