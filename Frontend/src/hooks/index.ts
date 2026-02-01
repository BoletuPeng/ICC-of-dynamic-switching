/**
 * Custom Hooks
 *
 * Reusable hooks for common patterns throughout the application.
 */

export { useFileOperations } from './useFileOperations';
export { useNodeDrag } from './useNodeDrag';
export {
  useShapeInference,
  useNodeShapeInfo,
  useConnectionInfo,
  useConnectionValidations,
} from './useShapeInference';
export type { NodeShapeInfo, ConnectionInfo, ShapeInferenceResult } from './useShapeInference';
