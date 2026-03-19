import { defineStore } from 'pinia';
import fsmData from '../../fsm.json';

// Initialize state from FSM signature schema
const initialState = {};

// Helper to set default values based on type
const getDefaultValue = (typeString) => {
  if (typeString.startsWith('array')) return [];
  if (typeString.startsWith('boolean')) return null; // Use null for uninitialized booleans as per FSM often checking for null
  if (typeString.startsWith('string')) return null;
  if (typeString.startsWith('number')) return 0;
  return null;
};

// Flatten all page signature schemas into one global state object
// The FSM defines signature_schema per page, but conceptually it's often a shared state or 
// at least we need to track it. For this implementation, we'll merge them.
fsmData.pages.forEach(page => {
  if (page.signature_schema) {
    Object.entries(page.signature_schema).forEach(([key, type]) => {
      if (!(key in initialState)) {
        initialState[key] = getDefaultValue(type);
      }
    });
  }
});

// Add internal tracking state
initialState.currentPageId = fsmData.meta.initial_page_id;

export const useSignatureStore = defineStore('signature', {
  state: () => ({ ...initialState }),
  actions: {
    setCurrentPageId(pageId) {
      this.currentPageId = pageId;
    },
    updateField(path, value) {
        // Simple dot notation update helper if needed
        // For JSONPath style updates, we might need a utility, 
        // but direct state access is preferred in Vue/Pinia
        if (path.startsWith('$.')) {
            const key = path.substring(2);
            this[key] = value;
        }
    }
  },
  persist: {
    storage: sessionStorage,
  },
});