/**
 * Calibration planner: recommend K and estimate costs.
 */

import type { CalibrationPlan, CalibrationPlanOptions } from '../types';

const PRICE_PER_MILLION: Record<string, number> = {
  'text-embedding-ada-002': 0.10,
  'text-embedding-3-small': 0.02,
  'text-embedding-3-large': 0.13,
  'voyage-3': 0.06,
  default: 0.05,
};

function estimateEmbedCost(
  modelId: string,
  nTexts: number,
  tokensPerText: number = 100,
): number {
  let price = PRICE_PER_MILLION.default;
  for (const [key, val] of Object.entries(PRICE_PER_MILLION)) {
    if (key !== 'default' && modelId.includes(key)) {
      price = val;
      break;
    }
  }
  const tokens = nTexts * tokensPerText;
  return (tokens / 1_000_000) * price;
}

/**
 * Recommend calibration size K from dims and corpus size.
 */
export function recommendK(corpusSize: number, dSrc: number, dTarget: number): number {
  const minDim = Math.min(dSrc, dTarget);
  const floor = 10 * minDim;
  let target = Math.max(floor, 2000);
  target = Math.min(target, 20000, Math.max(corpusSize, floor));
  return Math.round(target);
}

/**
 * Produce a calibration vs re-embed cost comparison.
 */
export function planCalibration(options: CalibrationPlanOptions): CalibrationPlan {
  const {
    corpusSize,
    sourceModel,
    targetModel,
    dSrc = 384,
    dTarget = 1024,
    tokensPerText = 100,
  } = options;

  const k = recommendK(corpusSize, dSrc, dTarget);
  const calCalls = 2 * k;
  const reembedCalls = corpusSize;
  const calUsd =
    estimateEmbedCost(sourceModel, k, tokensPerText) +
    estimateEmbedCost(targetModel, k, tokensPerText);
  const reembedUsd = estimateEmbedCost(targetModel, corpusSize, tokensPerText);

  return {
    corpusSize,
    recommendedK: k,
    sourceModel,
    targetModel,
    estCalibrationCalls: calCalls,
    estReembedCalls: reembedCalls,
    estCalibrationUsd: calUsd,
    estReembedUsd: reembedUsd,
    notes: [
      'Costs are rough public-list estimates, not invoices.',
      'Prefer in-domain sampling from your corpus over the generic set.',
      'Isotrieve cannot beat true re-embedding; run the quality gate before migrating.',
    ],
  };
}
