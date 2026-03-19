<template>
  <div class="min-h-screen bg-gray-100 pb-10">
    <!-- Header (simplified) -->
    <header class="bg-white shadow-sm sticky top-0 z-20 h-16 flex items-center px-4">
      <button 
        id="back-to-friends"
        @click="goBackFriends"
        class="flex items-center gap-2 text-gray-600 hover:bg-gray-100 px-3 py-2 rounded-lg transition-colors"
      >
        <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        <span class="font-medium">Back to Friends</span>
      </button>
    </header>

    <div class="max-w-5xl mx-auto bg-white shadow-sm rounded-b-xl overflow-hidden">
      <!-- Cover Photo -->
      <div class="h-64 md:h-80 bg-gray-300 relative">
        <img
            src="/images/UserCover.jpg"
            class="w-full h-full object-cover"
            alt="Cover"
        />
      </div>

      <!-- Profile Section -->
      <div class="px-4 pb-4 md:px-8 relative">
        <div class="flex flex-col md:flex-row items-center md:items-end -mt-16 md:-mt-12 gap-4">
          <!-- Avatar -->
          <div class="h-32 w-32 md:h-40 md:w-40 rounded-full border-4 border-white bg-white overflow-hidden shadow-md relative z-10">
            <img :src="profile?.avatar || '/images/photo1765160981.jpg'" class="w-full h-full object-cover" alt="Avatar" />
          </div>
          
          <!-- Name & Friends count -->
          <div class="flex-1 text-center md:text-left mb-2 md:mb-4">
            <h1 class="text-3xl font-bold text-gray-900">{{ profile?.name || 'User Name' }}</h1>
            <p class="text-gray-500 font-medium">{{ profile?.mutual || 0 }} mutual friends</p>
          </div>
          
          <!-- Actions -->
          <div class="flex gap-2 mb-4">
            <button 
              id="add-friend-button"
              @click="goToAddFriend"
              class="px-4 py-2 bg-blue-600 text-white font-semibold rounded-md shadow-sm hover:bg-blue-700 transition-colors flex items-center gap-2"
            >
              <svg class="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M8 9a3 3 0 100-6 3 3 0 000 6zM8 11a6 6 0 016 6H2a6 6 0 016-6zM16 7a1 1 0 10-2 0 1 1 0 002 0z" />
              </svg>
              Add Friend
            </button>
            <button 
              id="message-button"
              @click="goToMessage"
              class="px-4 py-2 bg-gray-200 text-gray-800 font-semibold rounded-md hover:bg-gray-300 transition-colors flex items-center gap-2"
            >
              <svg class="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clip-rule="evenodd" />
              </svg>
              Message
            </button>
          </div>
        </div>

        <!-- Divider -->
        <hr class="mt-6 mb-1 border-gray-300" />

        <!-- Tabs -->
        <div class="flex gap-1">
          <button class="px-4 py-3 text-blue-600 font-semibold border-b-2 border-blue-600">Posts</button>
          <button 
            id="profile-about-tab"
            @click="goToAbout"
            class="px-4 py-3 text-gray-600 font-semibold hover:bg-gray-100 rounded-lg transition-colors"
          >
            About
          </button>
          <button class="px-4 py-3 text-gray-600 font-semibold hover:bg-gray-100 rounded-lg transition-colors">Friends</button>
          <button class="px-4 py-3 text-gray-600 font-semibold hover:bg-gray-100 rounded-lg transition-colors">Photos</button>
        </div>
      </div>
    </div>

    <!-- Timeline Content (Placeholder) -->
    <div class="max-w-5xl mx-auto px-4 mt-4 grid grid-cols-1 md:grid-cols-5 gap-4">
      <div class="md:col-span-2 space-y-4">
        <div class="bg-white rounded-lg shadow-sm p-4">
          <h2 class="font-bold text-xl mb-3">Intro</h2>
          <p class="text-center text-gray-500 py-4">No intro details available.</p>
        </div>
        <div class="bg-white rounded-lg shadow-sm p-4">
          <h2 class="font-bold text-xl mb-3">Photos</h2>
          <div class="grid grid-cols-3 gap-1">
             <div class="bg-gray-200 aspect-square rounded-sm"></div>
             <div class="bg-gray-200 aspect-square rounded-sm"></div>
             <div class="bg-gray-200 aspect-square rounded-sm"></div>
          </div>
        </div>
      </div>
      <div class="md:col-span-3">
        <div class="bg-white rounded-lg shadow-sm p-4 text-center text-gray-500">
            No posts to show.
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'PROFILE_TIMELINE',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const profile = computed(() => {
      const id = route.params.id || signatureStore.selected_user_id;
      return [...dataStore.friends, ...dataStore.suggestedFriends].find(u => u.id === id);
    });
    
    onMounted(() => {
        if (!profile.value && route.params.id) {
            signatureStore.selected_user_id = route.params.id
        }
    })

    const goToAbout = async () => {
      signatureStore.currentPageId = 'PROFILE_ABOUT';
      await router.push({ name: 'PROFILE_ABOUT', params: { id: route.params.id } });
    };

    const goToAddFriend = async () => {
      signatureStore.currentPageId = 'FRIEND_REQUEST_CONFIRM';
      await router.push({ name: 'FRIEND_REQUEST_CONFIRM', params: { id: route.params.id } });
    };

    const goToMessage = async () => {
      signatureStore.currentPageId = 'MESSAGE_COMPOSE';
      await router.push({ name: 'MESSAGE_COMPOSE' });
    };

    const goBackFriends = async () => {
      signatureStore.currentPageId = 'FRIENDS_LIST';
      await router.push({ name: 'FRIENDS_LIST' });
    };

    return {
      profile,
      goToAbout,
      goToAddFriend,
      goToMessage,
      goBackFriends
    };
  }
}
</script>