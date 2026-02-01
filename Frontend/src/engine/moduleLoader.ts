/**
 * Module Loader - Dynamic loading of module definitions
 *
 * Uses Vite's import.meta.glob to automatically discover and load
 * all module definitions from the modules directory.
 */

import type { ModuleDefinition, ModulePortDefinition, ModuleParameterDefinition } from './types';
import type { NodeCategory } from '../types';

// =============================================================================
// Module Registry
// =============================================================================

/**
 * Global registry of loaded modules
 */
class ModuleRegistry {
  private modules: Map<string, ModuleDefinition> = new Map();
  private modulesByCategory: Map<string, ModuleDefinition[]> = new Map();
  private initialized = false;

  /**
   * Register a module definition
   */
  register(module: ModuleDefinition): void {
    this.modules.set(module.id, module);

    // Add to category map
    const category = module.category;
    if (!this.modulesByCategory.has(category)) {
      this.modulesByCategory.set(category, []);
    }
    this.modulesByCategory.get(category)!.push(module);
  }

  /**
   * Get a module by ID
   */
  get(id: string): ModuleDefinition | undefined {
    return this.modules.get(id);
  }

  /**
   * Get all modules
   */
  getAll(): ModuleDefinition[] {
    return Array.from(this.modules.values());
  }

  /**
   * Get modules by category
   */
  getByCategory(category: string): ModuleDefinition[] {
    return this.modulesByCategory.get(category) || [];
  }

  /**
   * Get all categories
   */
  getCategories(): string[] {
    return Array.from(this.modulesByCategory.keys());
  }

  /**
   * Check if initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Mark as initialized
   */
  setInitialized(): void {
    this.initialized = true;
  }

  /**
   * Clear all modules (for testing)
   */
  clear(): void {
    this.modules.clear();
    this.modulesByCategory.clear();
    this.initialized = false;
  }
}

export const moduleRegistry = new ModuleRegistry();

// =============================================================================
// Module Validation
// =============================================================================

/**
 * Validate a module definition structure
 */
export function validateModuleDefinition(module: unknown): module is ModuleDefinition {
  if (!module || typeof module !== 'object') return false;

  const m = module as Record<string, unknown>;

  // Required fields
  if (typeof m.id !== 'string' || !m.id) return false;
  if (typeof m.name !== 'string' || !m.name) return false;
  if (typeof m.category !== 'string' || !m.category) return false;
  if (typeof m.color !== 'string' || !m.color) return false;
  if (typeof m.icon !== 'string' || !m.icon) return false;

  // Validate ports
  if (!Array.isArray(m.inputs) || !Array.isArray(m.outputs)) return false;

  for (const port of [...(m.inputs as unknown[]), ...(m.outputs as unknown[])]) {
    if (!validatePortDefinition(port)) return false;
  }

  // Validate parameters
  if (!Array.isArray(m.parameters)) return false;

  return true;
}

/**
 * Validate a port definition
 */
function validatePortDefinition(port: unknown): port is ModulePortDefinition {
  if (!port || typeof port !== 'object') return false;

  const p = port as Record<string, unknown>;

  if (typeof p.id !== 'string' || !p.id) return false;
  if (typeof p.name !== 'string' || !p.name) return false;
  if (!Array.isArray(p.shape)) return false;

  return true;
}

// =============================================================================
// Dynamic Module Loading
// =============================================================================

/**
 * Load all modules from the modules directory
 * Uses Vite's import.meta.glob for automatic discovery
 */
export async function loadModules(): Promise<void> {
  if (moduleRegistry.isInitialized()) {
    return;
  }

  // Use Vite's glob import to find all module JSON files
  // This will be processed at build time
  const moduleFiles = import.meta.glob('../modules/*.json', { eager: true });

  for (const [path, moduleData] of Object.entries(moduleFiles)) {
    try {
      const module = (moduleData as { default: unknown }).default || moduleData;

      if (validateModuleDefinition(module)) {
        moduleRegistry.register(module);
        console.log(`[ModuleLoader] Loaded module: ${module.id}`);
      } else {
        console.warn(`[ModuleLoader] Invalid module definition in ${path}`);
      }
    } catch (error) {
      console.error(`[ModuleLoader] Error loading module from ${path}:`, error);
    }
  }

  moduleRegistry.setInitialized();
  console.log(`[ModuleLoader] Loaded ${moduleRegistry.getAll().length} modules`);
}

/**
 * Synchronously load modules from pre-imported data
 * Used as fallback or for testing
 */
export function loadModulesSync(modules: ModuleDefinition[]): void {
  for (const module of modules) {
    if (validateModuleDefinition(module)) {
      moduleRegistry.register(module);
    }
  }
  moduleRegistry.setInitialized();
}

// =============================================================================
// Module Access Helpers
// =============================================================================

/**
 * Get a module definition by ID
 */
export function getModule(id: string): ModuleDefinition | undefined {
  return moduleRegistry.get(id);
}

/**
 * Get all loaded modules
 */
export function getAllModules(): ModuleDefinition[] {
  return moduleRegistry.getAll();
}

/**
 * Get modules grouped by category
 */
export function getModulesByCategory(): Record<string, ModuleDefinition[]> {
  const result: Record<string, ModuleDefinition[]> = {};
  for (const category of moduleRegistry.getCategories()) {
    result[category] = moduleRegistry.getByCategory(category);
  }
  return result;
}

// =============================================================================
// Conversion Helpers
// =============================================================================

/**
 * Convert a ModuleDefinition to the legacy NodeDefinition format
 * for compatibility with existing UI components
 */
export function toLegacyNodeDefinition(module: ModuleDefinition): {
  id: string;
  name: string;
  description: string;
  category: NodeCategory;
  color: string;
  icon: string;
  inputs: ModulePortDefinition[];
  outputs: ModulePortDefinition[];
  parameters: ModuleParameterDefinition[];
} {
  return {
    id: module.id,
    name: module.name,
    description: module.description || '',
    category: module.category as NodeCategory,
    color: module.color,
    icon: module.icon,
    inputs: module.inputs,
    outputs: module.outputs,
    parameters: module.parameters,
  };
}

/**
 * Get all modules in legacy format
 */
export function getAllModulesLegacy() {
  return moduleRegistry.getAll().map(toLegacyNodeDefinition);
}
