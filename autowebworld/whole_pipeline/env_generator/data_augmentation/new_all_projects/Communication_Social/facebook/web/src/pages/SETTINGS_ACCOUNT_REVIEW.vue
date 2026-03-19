<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-lg overflow-hidden flex flex-col">
      <!-- Header -->
      <div class="relative px-4 py-3 border-b border-gray-100 flex items-center justify-center bg-white z-10">
        <div 
          id="settings-back-edit"
          @click="goBackEdit"
          class="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
        >
          <svg class="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
        </div>
        <h2 class="text-lg font-bold text-gray-900">Review Changes</h2>
        <div 
          id="settings-cancel-from-review" 
          @click="cancelReview"
          class="absolute right-4 top-1/2 transform -translate-y-1/2 p-1 text-gray-500 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
        >
          Cancel
        </div>
      </div>

      <!-- Content -->
      <div class="flex-1 p-6">
        <h3 class="font-medium text-gray-900 mb-4">You are about to save:</h3>
        
        <div class="bg-gray-50 rounded-lg border border-gray-200 divide-y divide-gray-200">
           <div class="p-4 flex justify-between items-center">
              <span class="text-gray-600">Name</span>
              <span class="font-semibold text-gray-900">Alex Johnson</span>
           </div>
           <div class="p-4 flex justify-between items-center">
              <span class="text-gray-600">Privacy</span>
              <span class="font-semibold text-gray-900 capitalize">{{ privacy?.replace('_', ' ') }}</span>
           </div>
        </div>
        
        <div class="mt-6 bg-blue-50 text-blue-800 text-sm p-4 rounded-lg">
           These changes will apply immediately to your account.
        </div>
      </div>

      <!-- Footer -->
      <div class="p-4 border-t border-gray-100 bg-gray-50">
        <button 
          id="settings-save-button"
          @click="saveSettings"
          class="w-full py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-sm hover:bg-blue-700 transition-colors"
        >
          Save Changes
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'SETTINGS_ACCOUNT_REVIEW',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    const privacy = computed(() => signatureStore.privacy_option);

    const saveSettings = async () => {
      signatureStore.currentPageId = 'ACCOUNT_SETTINGS_SAVED_SUCCESS';
      await router.push({ name: 'ACCOUNT_SETTINGS_SAVED_SUCCESS' });
    };

    const goBackEdit = async () => {
      signatureStore.currentPageId = 'SETTINGS_ACCOUNT';
      await router.push({ name: 'SETTINGS_ACCOUNT' });
    };

    const cancelReview = async () => {
      // Clear
      signatureStore.name_input_filled = null;
      signatureStore.privacy_option = null;
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    return {
      privacy,
      saveSettings,
      goBackEdit,
      cancelReview
    };
  }
}
</script>