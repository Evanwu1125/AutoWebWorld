<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center py-10 px-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-lg overflow-hidden">
      <!-- Header -->
      <div class="relative px-4 py-3 border-b border-gray-100 flex items-center justify-center">
        <h2 class="text-xl font-bold text-gray-900">Create Post</h2>
        <div 
          id="post-cancel" 
          @click="goBack"
          class="absolute right-4 top-1/2 transform -translate-y-1/2 p-2 bg-gray-100 rounded-full hover:bg-gray-200 cursor-pointer transition-colors"
        >
          <svg class="h-5 w-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
      </div>

      <!-- User Info & Audience -->
      <div class="p-4 flex items-center gap-3">
        <img class="h-10 w-10 rounded-full" src="/images/photo1765160844.jpg" alt="User" />
        <div>
          <h3 class="font-semibold text-gray-900 text-sm">Alex Johnson</h3>
          
          <!-- Audience Dropdown -->
          <div class="relative">
            <button 
              id="post-audience-dropdown"
              @click="toggleAudience"
              class="flex items-center gap-1 bg-gray-100 hover:bg-gray-200 px-2 py-1 rounded-md text-xs font-medium text-gray-700 transition-colors"
            >
              <span>{{ audienceLabel }}</span>
              <svg class="h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
              </svg>
            </button>
            
            <div v-if="audienceOpen" class="absolute top-full left-0 mt-1 w-48 bg-white rounded-md shadow-xl py-1 z-50 ring-1 ring-black ring-opacity-5">
              <div 
                id="audience-option-public"
                @click="selectAudience('public')"
                class="flex items-center gap-3 px-4 py-2 hover:bg-gray-100 cursor-pointer"
              >
                <span class="text-lg">🌍</span>
                <div class="flex flex-col">
                  <span class="text-sm font-medium">Public</span>
                  <span class="text-xs text-gray-500">Anyone on or off Facebook</span>
                </div>
              </div>
              <div 
                id="audience-option-friends"
                @click="selectAudience('friends')"
                class="flex items-center gap-3 px-4 py-2 hover:bg-gray-100 cursor-pointer"
              >
                <span class="text-lg">👥</span>
                <div class="flex flex-col">
                  <span class="text-sm font-medium">Friends</span>
                  <span class="text-xs text-gray-500">Your friends on Facebook</span>
                </div>
              </div>
              <div 
                id="audience-option-only-me"
                @click="selectAudience('only_me')"
                class="flex items-center gap-3 px-4 py-2 hover:bg-gray-100 cursor-pointer"
              >
                <span class="text-lg">🔒</span>
                <div class="flex flex-col">
                  <span class="text-sm font-medium">Only me</span>
                  <span class="text-xs text-gray-500">Only you can see this</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Text Input -->
      <div class="px-4 py-2">
        <textarea 
          id="post-composer-textarea"
          v-model="postText"
          @input="handleInput"
          placeholder="What's on your mind, Alex?"
          class="w-full h-40 resize-none border-none focus:ring-0 text-lg placeholder-gray-500"
        ></textarea>
      </div>
      
      <!-- Background Options (Visual Only) -->
      <div class="px-4 mb-4 flex items-center justify-between">
        <img src="/images/Backgrounds.jpg" class="h-9 w-9 cursor-pointer" alt="Backgrounds" />
        <span class="text-2xl text-gray-300 cursor-pointer">😊</span>
      </div>

      <!-- Add to Post -->
      <div class="px-4 pb-4">
        <div class="border border-gray-200 rounded-lg p-3 flex items-center justify-between shadow-sm">
          <span class="font-medium text-sm text-gray-900">Add to your post</span>
          <div class="flex gap-4">
            <span class="text-green-500 text-xl cursor-pointer hover:bg-gray-100 rounded-full p-1">🖼️</span>
            <span class="text-blue-500 text-xl cursor-pointer hover:bg-gray-100 rounded-full p-1">👤</span>
            <span class="text-yellow-500 text-xl cursor-pointer hover:bg-gray-100 rounded-full p-1">😊</span>
            <span class="text-red-500 text-xl cursor-pointer hover:bg-gray-100 rounded-full p-1">📍</span>
            <span class="text-gray-500 text-xl cursor-pointer hover:bg-gray-100 rounded-full p-1">...</span>
          </div>
        </div>
      </div>

      <!-- Footer Action -->
      <div class="p-4 border-t border-gray-100">
        <button 
          id="post-next-review"
          @click="goToReview"
          :disabled="!canProceed"
          class="w-full py-2 bg-blue-600 text-white font-semibold rounded-md shadow-sm hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          Next
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'CREATE_POST',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    // Initialize with existing state if any (for back navigation)
    const postText = ref(signatureStore.post_text || '');
    const audience = ref(signatureStore.post_audience || '');
    const audienceOpen = ref(false);

    const audienceLabel = computed(() => {
      switch(audience.value) {
        case 'public': return 'Public';
        case 'friends': return 'Friends';
        case 'only_me': return 'Only me';
        default: return 'Select Audience';
      }
    });

    const canProceed = computed(() => {
      return postText.value.length > 0 && audience.value.length > 0;
    });

    const toggleAudience = () => {
      audienceOpen.value = !audienceOpen.value;
    };

    const selectAudience = (value) => {
      audience.value = value;
      signatureStore.post_audience = value;
      audienceOpen.value = false;
    };

    const handleInput = () => {
      // For FSM "typed" logic, but storing actual text is better for UX
      signatureStore.post_text = postText.value; 
    };

    const goToReview = async () => {
      if (canProceed.value) {
        signatureStore.currentPageId = 'CREATE_POST_REVIEW';
        await router.push({ name: 'CREATE_POST_REVIEW' });
      }
    };

    const goBack = async () => {
      // Clear state on cancel
      signatureStore.post_text = null;
      signatureStore.post_audience = null;
      signatureStore.currentPageId = 'NEWS_FEED';
      await router.push({ name: 'NEWS_FEED' });
    };

    return {
      postText,
      audienceOpen,
      audienceLabel,
      canProceed,
      toggleAudience,
      selectAudience,
      handleInput,
      goToReview,
      goBack
    };
  }
}
</script>