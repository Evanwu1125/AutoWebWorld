<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-sm text-center p-8">
      <div class="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-green-100 mb-6">
        <svg class="h-10 w-10 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Request Sent!</h2>
      <p class="text-gray-600 mb-8">Your friend request has been sent successfully.</p>
      
      <div class="space-y-3">
        <button 
          id="friend-request-success-view-profile"
          @click="viewProfile"
          class="w-full py-2 bg-blue-600 text-white font-semibold rounded-md shadow-sm hover:bg-blue-700 transition-colors"
        >
          View Profile
        </button>
        <button 
          id="friend-request-success-home"
          @click="goHome"
          class="w-full py-2 bg-white text-gray-700 font-semibold rounded-md border border-gray-300 hover:bg-gray-50 transition-colors"
        >
          Go Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'FRIEND_REQUEST_SENT_SUCCESS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const viewProfile = async () => {
      const id = signatureStore.selected_user_id;
      signatureStore.currentPageId = 'PROFILE_TIMELINE';
      await router.push({ name: 'PROFILE_TIMELINE', params: { id } });
    };

    const goHome = async () => {
      signatureStore.success_message = "Friend request sent"; // FSM Effect
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    return {
      viewProfile,
      goHome
    };
  }
}
</script>