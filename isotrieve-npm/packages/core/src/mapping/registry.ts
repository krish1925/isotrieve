import { Mapping } from './base';
import { readIsotrieveHeader, readIsotrievePayload } from './format';
import { TypedMatrix } from '../math/matrix';
import type { ValidationReport } from '../types';

const REGISTRY = new Map<string, typeof Mapping>();

export function registerMapping(cls: typeof Mapping): void {
  REGISTRY.set(cls.mappingType, cls);
}

export function getMappingClass(type: string): typeof Mapping {
  ensureBuiltins();
  const cls = REGISTRY.get(type);
  if (!cls) {
    throw new Error(`Unknown mapping_type: "${type}". Registered types: ${[...REGISTRY.keys()].join(', ')}`);
  }
  return cls;
}

let _builtinsRegistered = false;
function ensureBuiltins(): void {
  if (_builtinsRegistered) return;
  _builtinsRegistered = true;

  const { RidgeMapping } = require('./ridge');
  const { OrthogonalProcrustesMapping, ProcrustesDiagMapping } = require('./procrustes');
  const { LowRankAffineMapping } = require('./lowrank');

  registerMapping(RidgeMapping);
  registerMapping(OrthogonalProcrustesMapping);
  registerMapping(ProcrustesDiagMapping);
  registerMapping(LowRankAffineMapping);
}

export function loadMapping(path: string): Mapping {
  ensureBuiltins();

  const { header, matrices } = readIsotrievePayload(path);
  const mappingType = header.mappingType as string;
  const cls = getMappingClass(mappingType);

  const instance = Object.create(cls.prototype) as Mapping;

  (instance as any)._fitted = true;
  (instance as any)._W = matrices.get('forward') ?? null;
  (instance as any)._WInv = matrices.get('inverse') ?? null;
  (instance as any)._dSrc = header.dSrc as number;
  (instance as any)._dTarget = header.dTarget as number;
  (instance as any)._bias = (header.bias as boolean) ?? false;
  (instance as any)._seed = (header.seed as number) ?? 0;
  (instance as any)._meta = (header.meta as Record<string, unknown>) ?? {};
  (instance as any)._validationReport = null;
  (instance as any)._recalibrator = null;

  const val = header.validation as ValidationReport | null;
  if (val !== null) {
    (instance as any)._validationReport = val;
  }

  const extraKeys = [...matrices.keys()].filter(
    (k) => k !== 'forward' && k !== 'inverse',
  );
  if (extraKeys.length > 0) {
    const extrasMap = new Map(extraKeys.map((k) => [k, matrices.get(k)!]));
    const proto = Object.getPrototypeOf(instance);
    if (proto && typeof proto._restoreExtraMatrices === 'function') {
      proto._restoreExtraMatrices.call(instance, extrasMap);
    }
  }

  if (mappingType === 'ridge' || mappingType === 'lowrank_affine') {
    const alpha = (val as any)?.alpha ?? null;
    (instance as any)._selectedAlpha = alpha;
    (instance as any)._rank = null;
    (instance as any)._normalizeOutput = true;
    (instance as any)._holdoutFraction = 0.1;
  } else if (mappingType === 'procrustes_diag') {
    (instance as any)._normalizeOutput = true;
    (instance as any)._holdoutFraction = 0.1;
    (instance as any)._scales = null;
    (instance as any)._scalesInv = null;
  } else if (mappingType === 'orthogonal_procrustes') {
    (instance as any)._normalizeOutput = true;
    (instance as any)._holdoutFraction = 0.1;
  }

  const recalData = header.scoreRecalV1;
  if (recalData !== undefined && recalData !== null) {
    try {
      const { ScoreRecalibrator } = require('../recalibration');
      (instance as any)._recalibrator = ScoreRecalibrator.fromJSON(recalData);
    } catch {
      // Recalibrator not available — skip silently
    }
  }

  return instance;
}
