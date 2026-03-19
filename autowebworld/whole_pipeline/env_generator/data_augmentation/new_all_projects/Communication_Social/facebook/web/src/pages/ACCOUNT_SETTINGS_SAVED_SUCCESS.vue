<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-sm text-center p-8">
      <div class="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-green-100 mb-6">
        <svg class="h-10 w-10 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Saved!</h2>
      <p class="text-gray-600 mb-8">Your account settings have been updated.</p>
      
      <div class="space-y-3">
        <button 
          id="settings-success-home"
          @click="goHome"
          class="w-full py-2 bg-blue-600 text-white font-semibold rounded-md shadow-sm hover:bg-blue-700 transition-colors"
        >
          Go Home
        </button>
        <button 
          id="settings-success-back-settings"
          @click="backToSettings"
          class="w-full py-2 bg-white text-gray-700 font-semibold rounded-md border border-gray-300 hover:bg-gray-50 transition-colors"
        >
          Back to Settings
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'ACCOUNT_SETTINGS_SAVED_SUCCESS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const goHome = async () => {
      signatureStore.success_message = "Settings saved"; // FSM Effect
      // Clear state
      signatureStore.name_input_filled = null;
      signatureStore.privacy_option = null;
      
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    const backToSettings = async () => {
      // Clear state
      signatureStore.name_input_filled = null;
      signatureStore.privacy_option = null;
      
      signatureStore.currentPageId = 'SETTINGS_ACCOUNT';
      await router.push({ name: 'SETTINGS_ACCOUNT' });
    };

    return {
      goHome,
      backToSettings
    };
  }
}
</script>