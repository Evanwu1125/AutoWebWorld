<template>
  <div class="min-h-screen bg-gray-900 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-2xl max-w-md w-full p-8 text-center">
      <div class="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10 text-blue-600 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
      </div>
      
      <h1 class="text-2xl font-bold text-gray-900 mb-2">Meeting Started</h1>
      <p class="text-gray-600 mb-8">{{ store.success_message || 'Your instant meeting is live.' }}</p>
      
      <div class="space-y-3">
        <button 
          id="instant-success-back-dashboard" 
          @click="goDashboard"
          class="w-full px-6 py-3 bg-blue-600 text-white rounded-lg font-bold hover:bg-blue-700 transition-colors shadow-md"
        >
          Back to Dashboard
        </button>
        
        <button 
          id="instant-success-go-home" 
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
  name: 'START_INSTANT_MEETING_SUCCESS',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const goDashboard = async () => {
      if (store.handleAction('ACT_INSTANT_SUCCESS_BACK_DASHBOARD')) {
        await router.push({ name: 'DASHBOARD' });
      }
    };

    const goHome = async () => {
      if (store.handleAction('ACT_INSTANT_SUCCESS_GO_HOME')) {
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