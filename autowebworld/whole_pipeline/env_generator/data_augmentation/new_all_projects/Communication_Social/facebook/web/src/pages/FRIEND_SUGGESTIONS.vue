<template>
  <div class="min-h-screen bg-gray-100 pb-10">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20 h-16 flex items-center px-4">
      <button 
        id="back-to-friends-list"
        @click="goBack"
        class="flex items-center gap-2 text-gray-600 hover:bg-gray-100 px-3 py-2 rounded-lg transition-colors"
      >
        <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        <span class="font-bold text-lg">Friend Suggestions</span>
      </button>
    </header>

    <div class="max-w-4xl mx-auto px-4 py-6">
      <h2 class="text-xl font-bold text-gray-900 mb-6">People You May Know</h2>
      
      <div id="friend-suggestions-list" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
        <div 
          v-for="person in suggestions" 
          :key="person.id" 
          class="bg-white rounded-lg shadow-sm overflow-hidden flex flex-col border border-gray-200"
          :class="{ 'suggested-visible': true }"
        >
          <div 
            class="h-48 w-full cursor-pointer relative"
            :class="`data-id-${person.id}`"
            @click="openProfile(person)"
          >
            <img :src="person.avatar" class="w-full h-full object-cover" :alt="person.name" />
          </div>
          
          <div class="p-3 flex flex-col flex-1">
            <h3 class="font-semibold text-gray-900 truncate cursor-pointer hover:underline" @click="openProfile(person)">
              {{ person.name }}
            </h3>
            <p class="text-sm text-gray-500 mb-3 flex-1">
              {{ person.mutual }} mutual friends
            </p>
            
            <div class="space-y-2 mt-auto">
              <button class="w-full py-2 bg-blue-100 text-blue-700 font-semibold rounded-md hover:bg-blue-200 transition-colors">
                Add Friend
              </button>
              <button class="w-full py-2 bg-gray-100 text-gray-700 font-semibold rounded-md hover:bg-gray-200 transition-colors">
                Remove
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'FRIEND_SUGGESTIONS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const suggestions = computed(() => dataStore.suggestedFriends);

    const openProfile = async (person) => {
      signatureStore.selected_user_id = person.id;
      // Clear anchor (FSM Effect)
      signatureStore.friend_suggestions_viewport_anchor_id = null;
      
      await router.push({ name: 'PROFILE_TIMELINE', params: { id: person.id } });
    };

    const goBack = async () => {
      signatureStore.currentPageId = 'FRIENDS_LIST';
      await router.push({ name: 'FRIENDS_LIST' });
    };

    return {
      suggestions,
      openProfile,
      goBack
    };
  }
}
</script>