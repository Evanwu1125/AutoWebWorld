<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg max-w-md w-full p-8 text-center">
      <div class="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-8 h-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
      </div>
      
      <h1 class="text-xl font-bold text-gray-900 mb-2">Profile Updated</h1>
      <p class="text-gray-600 mb-8">{{ store.success_message || 'Your profile name has been changed.' }}</p>
      
      <div class="space-y-3">
        <button 
          id="rename-success-back-profile" 
          @click="goBack"
          class="w-full px-6 py-2 bg-blue-600 text-white rounded-md font-medium hover:bg-blue-700 transition-colors"
        >
          Back to Profile
        </button>
        
        <button 
          id="rename-success-go-home" 
          @click="goHome"
          class="block w-full text-sm text-gray-500 hover:text-gray-800 mt-4"
        >
          Go to Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'RENAME_PROFILE_SUCCESS',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const goBack = async () => {
      if (store.handleAction('ACT_RENAME_SUCCESS_BACK_PROFILE')) {
        await router.push({ name: 'PROFILE_OVERVIEW' });
      }
    };

    const goHome = async () => {
      if (store.handleAction('ACT_RENAME_SUCCESS_GO_HOME')) {
        await router.push({ name: 'HOME' });
      }
    };

    return {
      store,
      goBack,
      goHome
    };
  }
}
</script>