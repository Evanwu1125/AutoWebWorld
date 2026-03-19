<template>
  <div v-if="isVisible" class="fixed inset-0 z-50 flex items-end justify-center px-4 py-6 pointer-events-none sm:p-6 sm:items-center sm:justify-center">
    <!-- Backdrop -->
    <div class="absolute inset-0 transition-opacity bg-gray-500 bg-opacity-75 backdrop-blur-sm" aria-hidden="true"></div>

    <!-- Modal Panel -->
    <div class="relative w-full max-w-md px-6 py-6 overflow-hidden text-left transition-all transform bg-white rounded-lg shadow-xl pointer-events-auto sm:my-8 sm:w-full sm:max-w-lg sm:p-6">
      <div>
        <div class="flex items-center justify-center w-12 h-12 mx-auto bg-blue-100 rounded-full">
          <span class="text-2xl">🍪</span>
        </div>
        <div class="mt-3 text-center sm:mt-5">
          <h3 class="text-lg font-medium leading-6 text-gray-900" id="modal-title">
            We Value Your Privacy
          </h3>
          <div class="mt-2">
            <p class="text-sm text-gray-500">
              We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
            </p>
          </div>
        </div>
      </div>
      <div class="mt-5 sm:mt-6 sm:grid sm:grid-cols-1 sm:gap-3 sm:grid-flow-row-dense">
        <button 
          type="button" 
          id="cookie-accept"
          class="inline-flex justify-center w-full px-4 py-2 text-base font-medium text-white bg-blue-600 border border-transparent rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:col-start-1 sm:text-sm"
          @click="acceptCookies"
        >
          Accept All
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
  name: 'CookieConsentModal',
  setup() {
    const isVisible = ref(false);
    const store = useSignatureStore();
    const { signature, currentPageId } = storeToRefs(store);
    const fsmRuntime = new FSMRuntime(fsmData, { store });

    const checkVisibility = () => {
      // Only show on HOME page if consent not given
      if (currentPageId.value === 'HOME' && signature.value.cookie_consent_given !== true) {
        isVisible.value = true;
      } else {
        isVisible.value = false;
      }
    };

    onMounted(() => {
      checkVisibility();
    });

    watch([currentPageId, () => signature.value.cookie_consent_given], () => {
      checkVisibility();
    });

    const acceptCookies = () => {
      // Find the action definition to execute effects
      const action = fsmData.pages.find(p => p.id === 'HOME').actions.find(a => a.id === 'ACT_HOME_ACCEPT_COOKIES');
      
      if (action) {
        const nextSig = fsmRuntime.applyEffects(action, signature.value);
        // Update store directly
        signature.value.cookie_consent_given = nextSig.cookie_consent_given;
        isVisible.value = false;
      }
    };

    return {
      isVisible,
      acceptCookies
    };
  }
}
</script>