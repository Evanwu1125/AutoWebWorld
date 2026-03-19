import { defineStore } from 'pinia';
import fsmData from '../../fsm.json';
import { FSMRuntime } from '../fsm';

// Initialize default state from FSM schema
const initializeState = () => {
  const state = {};
  
  // Global signature
  if (fsmData.meta.global_signature_schema) {
    Object.keys(fsmData.meta.global_signature_schema).forEach(key => {
      state[key] = null; // Default to null or appropriate empty value
      if (fsmData.meta.global_signature_schema[key].startsWith('array')) {
          state[key] = [];
      }
    });
  }

  // Page-specific signatures
  fsmData.pages.forEach(page => {
    if (page.signature_schema) {
      Object.keys(page.signature_schema).forEach(key => {
        if (state[key] === undefined) {
             state[key] = null;
             if (page.signature_schema[key].startsWith('array')) {
                state[key] = [];
             }
        }
      });
    }
  });

  return state;
};

export const useSignatureStore = defineStore('signature', {
  state: () => ({
    ...initializeState(),
    currentPageId: fsmData.meta.initial_page_id,
    // Add internal state for tracking
    _location_permission_denied: false, 
  }),
  
  getters: {
    // Helper to get raw signature object for FSM operations
    getSignature(state) {
        const { currentPageId, _location_permission_denied, ...signature } = state;
        return signature;
    }
  },

  actions: {
    setCurrentPageId(pageId) {
      this.currentPageId = pageId;
    },

    // Generic action handler
    handleAction(actionId, parameters = {}) {
      const fsmRuntime = new FSMRuntime(fsmData, { currentPageId: this.currentPageId });
      
      // Find the action definition
      let actionDef = null;
      let pageDef = fsmData.pages.find(p => p.id === this.currentPageId);
      
      if (pageDef) {
        actionDef = pageDef.actions.find(a => a.id === actionId);
      }

      if (!actionDef) {
        console.error(`Action ${actionId} not found on page ${this.currentPageId}`);
        return;
      }

      // Check preconditions
      const currentSignature = this.getSignature;
      const canExecute = fsmRuntime.checkPreconditions(actionDef.preconditions, currentSignature);
      
      if (!canExecute) {
        console.warn(`Preconditions failed for action ${actionId}`);
        return;
      }

      // Apply effects
      const newSignature = fsmRuntime.applyEffects(actionDef.effects, currentSignature, parameters);
      
      // Update store state with new signature
      Object.keys(newSignature).forEach(key => {
        this[key] = newSignature[key];
      });

      return actionDef;
    }
  },
  
  persist: {
    storage: sessionStorage,
  },
});