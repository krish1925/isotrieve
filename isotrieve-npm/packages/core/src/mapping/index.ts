/**
 * Mapping module barrel export.
 */

export { Mapping, type ApplyMappingOptions } from './base';
export { RidgeMapping, type RidgeMappingOptions } from './ridge';
export { OrthogonalProcrustesMapping, ProcrustesDiagMapping } from './procrustes';
export { LowRankAffineMapping, type LowRankAffineMappingOptions } from './lowrank';
export { ExternalMapping, type ExternalTransformFn } from './external';
export {
  registerMapping,
  getMappingClass,
  loadMapping,
} from './registry';
export {
  writeIsotrieveFile,
  readIsotrieveHeader,
  readIsotrievePayload,
  ISOTRIEVE_MAGIC,
  FORMAT_VERSION,
} from './format';
