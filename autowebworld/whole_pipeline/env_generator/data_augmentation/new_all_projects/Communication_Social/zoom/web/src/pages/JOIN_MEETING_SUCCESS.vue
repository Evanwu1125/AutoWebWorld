<template>
  <div class="min-h-screen bg-gray-900 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-2xl max-w-md w-full p-8 text-center">
      <div class="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
      </div>
      
      <h1 class="text-2xl font-bold text-gray-900 mb-2">Connected!</h1>
      <p class="text-gray-600 mb-8">{{ store.success_message || 'You have joined the meeting.' }}</p>
      
      <div class="space-y-3">
        <button 
          id="join-success-back-dashboard" 
          @click="goDashboard"
          class="w-full px-6 py-3 bg-blue-600 text-white rounded-lg font-bold hover:bg-blue-700 transition-colors shadow-md"
        >
          Return to Dashboard
        </button>
        
        <button 
          id="join-success-go-home" 
          @click="goHome"
          class="block w-full text-sm text-gray-500 hover:text-gray-800 mt-4 underline"
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
  name: 'JOIN_MEETING_SUCCESS',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const goDashboard = async () => {
      if (store.handleAction('ACT_JOIN_SUCCESS_BACK_DASHBOARD')) {
        await router.push({ name: 'DASHBOARD' });
      }
    };

    const goHome = async () => {
      if (store.handleAction('ACT_JOIN_SUCCESS_GO_HOME')) {
        await router.push({ name: 'HOME' });
      }
    };

    return {
      store,
      goDashboard,
      goHome
    };
  }
}
</script>