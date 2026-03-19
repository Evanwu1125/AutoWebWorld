<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center py-10 px-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-lg overflow-hidden">
      <!-- Header -->
      <div class="relative px-4 py-3 border-b border-gray-100 flex items-center justify-center">
        <div 
          id="post-edit-back"
          @click="goBackEdit"
          class="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
        >
          <svg class="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
        </div>
        <h2 class="text-xl font-bold text-gray-900">Review Post</h2>
        <div 
          id="post-cancel-from-review" 
          @click="cancelReview"
          class="absolute right-4 top-1/2 transform -translate-y-1/2 p-2 bg-gray-100 rounded-full hover:bg-gray-200 cursor-pointer transition-colors"
        >
          <svg class="h-5 w-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
      </div>

      <!-- Preview Content -->
      <div class="p-4">
        <div class="flex items-center gap-3 mb-3">
          <img class="h-10 w-10 rounded-full" src="/images/photo1765160852.jpg" alt="User" />
          <div>
            <h3 class="font-semibold text-gray-900 text-sm">Alex Johnson</h3>
            <div class="flex items-center gap-1 text-xs text-gray-500">
              <span>Just now</span>
              <span>•</span>
              <span v-if="audience === 'public'">🌍 Public</span>
              <span v-else-if="audience === 'friends'">👥 Friends</span>
              <span v-else>🔒 Only me</span>
            </div>
          </div>
        </div>
        
        <p class="text-gray-900 text-lg leading-relaxed mb-4">{{ postText }}</p>
        
        <div class="bg-gray-50 border border-gray-200 rounded-lg p-4 text-center text-gray-500 text-sm">
          (No photo/video attached)
        </div>
      </div>

      <!-- Warning/Info -->
      <div class="px-4 pb-4">
        <div class="bg-blue-50 text-blue-800 text-xs p-3 rounded-md flex items-start gap-2">
          <span class="text-lg">ℹ️</span>
          <p>This post will be visible to your selected audience immediately. Please review content against our Community Standards.</p>
        </div>
      </div>

      <!-- Footer Action -->
      <div class="p-4 border-t border-gray-100">
        <button 
          id="post-publish-button"
          @click="publishPost"
          class="w-full py-2 bg-blue-600 text-white font-semibold rounded-md shadow-sm hover:bg-blue-700 transition-colors"
        >
          Post
        </button>
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
  name: 'CREATE_POST_REVIEW',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();
    
    const postText = computed(() => signatureStore.post_text);
    const audience = computed(() => signatureStore.post_audience);

    const publishPost = async () => {
      // Create new post in mock data
      const newPost = {
        id: `post_new_${Date.now()}`,
        user_id: 'user_1',
        author_name: 'Alex Johnson',
        author_avatar: '/images/photo1765160852.jpg',
        content: postText.value,
        image: null,
        time: 'Just now',
        likes: 0,
        comments: 0
      };
      
      dataStore.posts.unshift(newPost); // Add to top of feed
      
      signatureStore.currentPageId = 'POST_PUBLISH_SUCCESS';
      await router.push({ name: 'POST_PUBLISH_SUCCESS' });
    };

    const goBackEdit = async () => {
      signatureStore.currentPageId = 'CREATE_POST';
      await router.push({ name: 'CREATE_POST' });
    };

    const cancelReview = async () => {
      // Clear state
      signatureStore.post_text = null;
      signatureStore.post_audience = null;
      signatureStore.currentPageId = 'NEWS_FEED';
      await router.push({ name: 'NEWS_FEED' });
    };

    return {
      postText,
      audience,
      publishPost,
      goBackEdit,
      cancelReview
    };
  }
}
</script>