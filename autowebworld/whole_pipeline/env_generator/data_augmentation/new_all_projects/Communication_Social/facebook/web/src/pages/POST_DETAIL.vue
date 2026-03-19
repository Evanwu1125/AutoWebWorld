<template>
  <div class="min-h-screen bg-gray-100 pb-10">
    <!-- Header (simplified) -->
    <header class="bg-white shadow-sm sticky top-0 z-20 h-16 flex items-center px-4">
      <button 
        id="back-to-feed"
        @click="goBack"
        class="flex items-center gap-2 text-gray-600 hover:bg-gray-100 px-3 py-2 rounded-lg transition-colors"
      >
        <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        <span class="font-medium">Back to News Feed</span>
      </button>
    </header>

    <!-- Post Content -->
    <div class="max-w-2xl mx-auto mt-6 px-4">
      <div v-if="post" class="bg-white rounded-lg shadow-md overflow-hidden">
        <!-- Author -->
        <div class="p-4 flex items-center justify-between border-b border-gray-100">
          <div class="flex items-center gap-3">
            <img :src="post.author_avatar" class="h-10 w-10 rounded-full" alt="Author" />
            <div>
              <h1 class="font-bold text-gray-900 text-base">{{ post.author_name }}</h1>
              <div class="text-xs text-gray-500 flex items-center gap-1">
                <span>{{ post.time }}</span>
                <span>•</span>
                <span>🌍</span>
              </div>
            </div>
          </div>
          <button class="text-gray-400 hover:bg-gray-100 p-2 rounded-full">
            <span class="text-xl">...</span>
          </button>
        </div>

        <!-- Body -->
        <div class="p-0">
          <p v-if="post.content" class="px-4 py-3 text-gray-900 text-lg leading-relaxed">{{ post.content }}</p>
          <img v-if="post.image" :src="post.image" class="w-full h-auto object-cover" alt="Post Content" />
        </div>

        <!-- Stats -->
        <div class="px-4 py-3 flex items-center justify-between text-gray-500 text-sm border-b border-gray-100">
          <div class="flex items-center gap-1">
            <span class="bg-blue-500 text-white p-1 rounded-full text-xs w-5 h-5 flex items-center justify-center">👍</span>
            <span class="font-medium">{{ post.likes }}</span>
          </div>
          <div class="flex gap-4">
            <span>{{ post.comments }} comments</span>
            <span>12 shares</span>
          </div>
        </div>

        <!-- Actions -->
        <div class="px-4 py-1 flex items-center justify-between border-b border-gray-100">
          <button class="flex-1 py-2 flex items-center justify-center gap-2 hover:bg-gray-50 rounded-md transition-colors text-gray-600 font-medium">
            <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
            </svg>
            Like
          </button>
          <button class="flex-1 py-2 flex items-center justify-center gap-2 hover:bg-gray-50 rounded-md transition-colors text-gray-600 font-medium">
            <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
            </svg>
            Comment
          </button>
          <button class="flex-1 py-2 flex items-center justify-center gap-2 hover:bg-gray-50 rounded-md transition-colors text-gray-600 font-medium">
            <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
            </svg>
            Share
          </button>
        </div>

        <!-- Mock Comments Section -->
        <div class="p-4 bg-gray-50">
          <p class="font-semibold text-gray-700 mb-4">Comments</p>
          <div class="flex gap-3 mb-4">
            <img src="/images/Comments.jpg" class="h-8 w-8 rounded-full" alt="Me" />
            <div class="flex-1 bg-white border border-gray-200 rounded-full px-4 py-2 text-gray-500 text-sm cursor-text">
              Write a comment...
            </div>
          </div>
          <!-- Placeholder comment -->
          <div class="flex gap-3">
            <div class="h-8 w-8 rounded-full bg-gray-300 flex-shrink-0"></div>
            <div>
              <div class="bg-white px-4 py-2 rounded-2xl shadow-sm inline-block">
                <span class="font-bold text-xs block">User Name</span>
                <span class="text-sm">This is a great post!</span>
              </div>
              <div class="text-xs text-gray-500 mt-1 ml-2 space-x-2">
                <span class="font-bold cursor-pointer hover:underline">Like</span>
                <span class="font-bold cursor-pointer hover:underline">Reply</span>
                <span>2h</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div v-else class="text-center py-20">
        <p class="text-gray-500 text-lg">Post not found.</p>
        <button @click="goBack" class="mt-4 text-blue-600 hover:underline">Go Back</button>
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
  name: 'POST_DETAIL',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const post = computed(() => {
      const id = route.params.id || signatureStore.selected_post_id;
      return dataStore.posts.find(p => p.id === id);
    });
    
    onMounted(() => {
        if (!post.value && route.params.id) {
            signatureStore.selected_post_id = route.params.id
        }
    })

    const goBack = async () => {
      signatureStore.currentPageId = 'NEWS_FEED';
      await router.push({ name: 'NEWS_FEED' });
    };

    return {
      post,
      goBack
    };
  }
}
</script>