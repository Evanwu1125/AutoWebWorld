<template>
  <div v-if="isVisible" class="fixed inset-0 z-50 flex items-end justify-center px-4 py-6 pointer-events-none sm:p-6 sm:items-center sm:justify-center">
    <!-- Backdrop -->
    <div class="absolute inset-0 transition-opacity bg-gray-500 bg-opacity-75 backdrop-blur-sm" aria-hidden="true"></div>

    <!-- Modal Panel -->
    <div class="relative w-full max-w-md px-6 py-6 overflow-hidden text-left transition-all transform bg-white rounded-lg shadow-xl pointer-events-auto sm:my-8 sm:w-full sm:max-w-lg sm:p-6">
      <div>
        <div class="flex items-center justify-center w-12 h-12 mx-auto bg-green-100 rounded-full">
          <svg class="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
        </div>
        <div class="mt-3 text-center sm:mt-5">
          <h3 class="text-lg font-medium leading-6 text-gray-900" id="modal-title">
            Location Permission Required
          </h3>
          <div class="mt-2">
            <p class="text-sm text-gray-500">
              This app needs access to your location to provide better service and local repository suggestions.
            </p>
          </div>
        </div>
      </div>
      <div class="mt-5 sm:mt-6 sm:grid sm:grid-cols-1 sm:gap-3 sm:grid-flow-row-dense">
        <button 
          type="button" 
          id="permission-location-allow"
          class="inline-flex justify-center w-full px-4 py-2 text-base font-medium text-white bg-green-600 border border-transparent rounded-md shadow-sm hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 sm:col-start-1 sm:text-sm"
          @click="allowPermission"
        >
          Allow
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, watch } from 'vue';
import { useSignatureStore } from '../stores/signature';
import { storeToRefs } from 'pinia';
import fsmData from '../../fsm.json';
import { FSMRuntime } from '../fsm/FSMRuntime';

export default {
  name: 'PermissionModal',
  setup() {
    const isVisible = ref(false);
    const store = useSignatureStore();
    const { signature, currentPageId } = storeToRefs(store);
    const fsmRuntime = new FSMRuntime(fsmData, { store });

    const checkVisibility = () => {
      // Check if current page requires location permission
      const pageDef = fsmData.pages.find(p => p.id === currentPageId.value);
      if (!pageDef) return;

      const hasPermissionField = pageDef.signature_schema && 'location_permission_granted' in pageDef.signature_schema;
      const notGranted = signature.value.location_permission_granted !== true;

      if (hasPermissionField && notGranted) {
        isVisible.value = true;
      } else {
        isVisible.value = false;
      }
    };

    onMounted(() => {
      checkVisibility();
    });

    watch([currentPageId, () => signature.value.location_permission_granted], () => {
      checkVisibility();
    });

    const allowPermission = () => {
      // Find the action definition - strictly speaking should find the action on the current page
      // But for simplicity we look for any action that grants this permission on current page
      const pageDef = fsmData.pages.find(p => p.id === currentPageId.value);
      if (pageDef) {
        const action = pageDef.actions.find(a => 
          a.effects && a.effects.some(e => e.path.includes('location_permission_granted') && e.value === true)
        );

        if (action) {
          const nextSig = fsmRuntime.applyEffects(action, signature.value);
          signature.value.location_permission_granted = nextSig.location_permission_granted;
          isVisible.value = false;
        } else {
           // Fallback if precise action not found easily, or just force set for UI dev
           signature.value.location_permission_granted = true;
           isVisible.value = false;
        }
      }
    };

    return {
      isVisible,
      allowPermission
    };
  }
}
</script>