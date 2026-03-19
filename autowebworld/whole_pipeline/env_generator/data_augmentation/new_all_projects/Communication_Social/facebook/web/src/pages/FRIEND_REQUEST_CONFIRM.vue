<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-sm overflow-hidden">
      <div class="p-6 text-center">
        <div class="h-24 w-24 mx-auto mb-4 rounded-full border-4 border-gray-100 overflow-hidden">
           <img :src="profile?.avatar || '/images/photo1765161047.jpg'" class="w-full h-full object-cover" alt="Avatar" />
        </div>
        
        <h2 class="text-xl font-bold text-gray-900 mb-2">Add {{ profile?.name || 'User' }}?</h2>
        <p class="text-gray-600 mb-6 text-sm">
          {{ profile?.name }} will receive a friend request from you.
        </p>
        
        <div class="space-y-3">
          <button 
            id="confirm-add-friend"
            @click="sendRequest"
            class="w-full py-2 bg-blue-600 text-white font-semibold rounded-md shadow-sm hover:bg-blue-700 transition-colors"
          >
            Send Request
          </button>
          <button 
            id="cancel-add-friend"
            @click="cancelRequest"
            class="w-full py-2 bg-white text-gray-700 font-semibold rounded-md border border-gray-300 hover:bg-gray-50 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'FRIEND_REQUEST_CONFIRM',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const profile = computed(() => {
      const id = route.params.id || signatureStore.selected_user_id;
      // Search in suggestions first as that's where we usually add friends from
      let user = dataStore.suggestedFriends.find(u => u.id === id);
      if (!user) {
         user = dataStore.friends.find(u => u.id === id);
      }
      return user;
    });

    const sendRequest = async () => {
      signatureStore.currentPageId = 'FRIEND_REQUEST_SENT_SUCCESS';
      await router.push({ name: 'FRIEND_REQUEST_SENT_SUCCESS' });
    };

    const cancelRequest = async () => {
      const id = route.params.id || signatureStore.selected_user_id;
      signatureStore.currentPageId = 'PROFILE_TIMELINE';
      await router.push({ name: 'PROFILE_TIMELINE', params: { id } });
    };

    return {
      profile,
      sendRequest,
      cancelRequest
    };
  }
}
</script>