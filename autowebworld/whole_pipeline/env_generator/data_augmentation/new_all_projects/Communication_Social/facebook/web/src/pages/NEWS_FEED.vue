<template>
  <div class="min-h-screen bg-gray-100">
    <PermissionModal />
    
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div class="flex items-center gap-2 cursor-pointer" id="header-logo-home" @click="goHome">
          <img class="h-8 w-8" src="/images/photo1765160784.jpg" alt="Logo" />
          <span class="font-bold text-2xl text-blue-600 hidden sm:block">facebook</span>
        </div>
        <div class="flex items-center gap-3">
          <div class="relative bg-gray-100 rounded-full px-3 py-2 w-10 h-10 flex items-center justify-center hover:bg-gray-200 cursor-pointer">
            <svg class="h-5 w-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          <img class="h-9 w-9 rounded-full cursor-pointer" src="/images/UserProfile.jpg" alt="Me" />
        </div>
      </div>
    </header>

    <div class="max-w-4xl mx-auto px-0 sm:px-4 py-6 grid grid-cols-1 md:grid-cols-3 gap-6">
      <!-- Left Sidebar (Shortcuts) -->
      <div class="hidden md:block col-span-1">
        <div class="sticky top-24 space-y-2">
          <div class="flex items-center gap-3 p-2 hover:bg-gray-200 rounded-lg cursor-pointer transition-colors">
            <img class="h-8 w-8 rounded-full" src="/images/UserProfile.jpg" alt="User" />
            <span class="font-medium text-sm">Alex Johnson</span>
          </div>
          <div class="flex items-center gap-3 p-2 hover:bg-gray-200 rounded-lg cursor-pointer transition-colors">
            <img class="h-8 w-8" src="/images/Friends.jpg" alt="Friends" />
            <span class="font-medium text-sm">Friends</span>
          </div>
          <div class="flex items-center gap-3 p-2 hover:bg-gray-200 rounded-lg cursor-pointer transition-colors">
            <img class="h-8 w-8" src="/images/Marketplace.jpg" alt="Marketplace" />
            <span class="font-medium text-sm">Marketplace</span>
          </div>
          <div class="flex items-center gap-3 p-2 hover:bg-gray-200 rounded-lg cursor-pointer transition-colors">
            <img class="h-8 w-8" src="/images/Groups.jpg" alt="Groups" />
            <span class="font-medium text-sm">Groups</span>
          </div>
        </div>
      </div>

      <!-- Main Feed -->
      <div class="col-span-1 md:col-span-2 space-y-4">
        <!-- Create Post Widget -->
        <div class="bg-white rounded-lg shadow-sm p-4">
          <div class="flex gap-3 mb-3">
            <img class="h-10 w-10 rounded-full" src="/images/UserProfile.jpg" alt="Me" />
            <div 
              id="create-post-cta" 
              @click="goToCreatePost"
              class="bg-gray-100 hover:bg-gray-200 rounded-full px-4 py-2 flex-grow flex items-center cursor-pointer transition-colors"
            >
              <span class="text-gray-500 text-sm">What's on your mind, Alex?</span>
            </div>
          </div>
          <div class="flex justify-between pt-3 border-t border-gray-100">
            <div class="flex items-center gap-2 px-2 py-1 hover:bg-gray-100 rounded-md cursor-pointer transition-colors">
              <span class="text-red-500 text-xl">📹</span>
              <span class="text-gray-600 text-sm font-medium">Live Video</span>
            </div>
            <div class="flex items-center gap-2 px-2 py-1 hover:bg-gray-100 rounded-md cursor-pointer transition-colors">
              <span class="text-green-500 text-xl">🖼️</span>
              <span class="text-gray-600 text-sm font-medium">Photo/Video</span>
            </div>
            <div class="flex items-center gap-2 px-2 py-1 hover:bg-gray-100 rounded-md cursor-pointer transition-colors">
              <span class="text-yellow-500 text-xl">😊</span>
              <span class="text-gray-600 text-sm font-medium">Feeling/Activity</span>
            </div>
          </div>
        </div>

        <!-- Filters Section -->
        <div class="bg-white rounded-lg shadow-sm p-4 space-y-4">
          <div class="flex items-center justify-between">
            <h3 class="font-semibold text-gray-900">Feed Filters</h3>
            
            <!-- Sort Dropdown -->
            <div class="relative">
              <button 
                id="news-feed-sort-dropdown" 
                @click="toggleSort"
                class="flex items-center gap-1 text-gray-600 bg-gray-100 px-3 py-1.5 rounded-md text-sm font-medium hover:bg-gray-200 transition-colors"
              >
                Sort: {{ sortOption === 'top_stories' ? 'Top Stories' : 'Most Recent' }}
                <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              <div v-if="sortOpen" class="absolute right-0 mt-2 w-40 bg-white rounded-md shadow-lg py-1 z-10 ring-1 ring-black ring-opacity-5">
                <div 
                  id="sort-option-top-stories"
                  @click="selectSort('top_stories')"
                  class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                >
                  Top Stories
                </div>
                <div 
                  id="sort-option-most-recent"
                  @click="selectSort('most_recent')"
                  class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                >
                  Most Recent
                </div>
              </div>
            </div>
          </div>
          
          <div class="flex flex-wrap items-center gap-6">
            <!-- Filter Checkbox -->
            <label class="flex items-center gap-2 cursor-pointer select-none">
              <div 
                id="filter-friends-only-checkbox"
                class="w-5 h-5 border-2 border-gray-300 rounded flex items-center justify-center transition-colors"
                :class="{ 'bg-blue-600 border-blue-600': filters.friendsOnly }"
                @click="toggleFriendsOnly"
              >
                <svg v-if="filters.friendsOnly" class="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <span class="text-sm font-medium text-gray-700">Friends Only</span>
            </label>
            
            <!-- Date Range Slider -->
            <div class="flex-1 min-w-[200px]">
              <div class="flex justify-between text-xs text-gray-500 mb-1">
                <span>All Time</span>
                <span>Past 24h</span>
              </div>
              <input 
                id="filter-date-range-slider"
                type="range" 
                min="0" 
                max="100" 
                step="25"
                v-model="filters.dateRange"
                @input="applyFilters"
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
            </div>
          </div>
        </div>

        <!-- Feed List -->
        <div id="news-feed-list" class="space-y-4">
          <div 
            v-for="post in filteredPosts" 
            :key="post.id" 
            class="bg-white rounded-lg shadow-sm overflow-hidden"
            :class="{ 'post-visible': true, 'post-filtered': isFiltered }"
          >
            <!-- Post Header -->
            <div class="p-4 flex items-center justify-between">
              <div class="flex items-center gap-3">
                <img :src="post.author_avatar" class="h-10 w-10 rounded-full" alt="Author" />
                <div>
                  <h4 class="font-bold text-gray-900 text-sm hover:underline cursor-pointer">{{ post.author_name }}</h4>
                  <p class="text-xs text-gray-500">{{ post.time }} • 🌍</p>
                </div>
              </div>
              <button class="text-gray-400 hover:bg-gray-100 p-2 rounded-full">
                <svg class="h-5 w-5" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M6 10a2 2 0 11-4 0 2 2 0 014 0zM12 10a2 2 0 11-4 0 2 2 0 014 0zM16 12a2 2 0 100-4 2 2 0 000 4z" />
                </svg>
              </button>
            </div>

            <!-- Post Content -->
            <div 
              class="cursor-pointer"
              :class="`data-id-${post.id}`"
              @click="openPost(post)"
            >
              <p v-if="post.content" class="px-4 pb-3 text-gray-800 text-sm leading-relaxed">{{ post.content }}</p>
              <img v-if="post.image" :src="post.image" class="w-full h-auto object-cover max-h-[500px]" alt="Post Content" />
            </div>

            <!-- Post Stats -->
            <div class="px-4 py-2 border-b border-gray-100 flex justify-between items-center text-xs text-gray-500">
              <div class="flex items-center gap-1">
                <span class="bg-blue-500 text-white p-0.5 rounded-full text-[10px] w-4 h-4 flex items-center justify-center">👍</span>
                <span>{{ post.likes }}</span>
              </div>
              <div class="flex gap-3">
                <span>{{ post.comments }} comments</span>
                <span>Share</span>
              </div>
            </div>

            <!-- Post Actions -->
            <div class="px-2 py-1 flex justify-between items-center">
              <button class="flex-1 flex items-center justify-center gap-2 py-2 hover:bg-gray-50 rounded-md transition-colors text-gray-600">
                <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5" />
                </svg>
                <span class="font-medium text-sm">Like</span>
              </button>
              <button class="flex-1 flex items-center justify-center gap-2 py-2 hover:bg-gray-50 rounded-md transition-colors text-gray-600">
                <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                </svg>
                <span class="font-medium text-sm">Comment</span>
              </button>
              <button class="flex-1 flex items-center justify-center gap-2 py-2 hover:bg-gray-50 rounded-md transition-colors text-gray-600">
                <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                </svg>
                <span class="font-medium text-sm">Share</span>
              </button>
            </div>
          </div>

          <!-- End of Feed -->
          <div v-if="filteredPosts.length === 0" class="text-center py-10 bg-white rounded-lg shadow-sm">
            <p class="text-gray-500">No posts found matching your filters.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';
import PermissionModal from '../components/PermissionModal.vue';
import { orderBy } from 'lodash-es';

export default {
  name: 'NEWS_FEED',
  components: {
    PermissionModal
  },
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const sortOpen = ref(false);
    const sortOption = ref('top_stories'); // Default sort
    const filters = ref({
      friendsOnly: false,
      dateRange: 0 // 0 = all time, 100 = very recent
    });

    // Check if any filter is active
    const isFiltered = computed(() => {
      return filters.value.friendsOnly || filters.value.dateRange > 0;
    });

    const filteredPosts = computed(() => {
      let result = [...dataStore.posts];

      // Apply Friends Only Filter
      if (filters.value.friendsOnly) {
        // Mock logic: assume even IDs are friends
        // In real app, check if user_id in friends list
        const friendIds = dataStore.friends.map(f => f.id);
        result = result.filter(p => friendIds.includes(p.user_id));
      }

      // Apply Date Range Filter (Mocked based on slider value)
      // Slider 0 = All Time, Slider > 50 = Recent (e.g., < 2 days)
      if (filters.value.dateRange > 50) {
        result = result.filter(p => p.time.includes('h') || p.time === '1d');
      }

      // Apply Sort
      if (sortOption.value === 'most_recent') {
        // Mock sort by "time" (string logic approximation or original order)
        // For mock, let's reverse to show "newest" first if original is chronological
        // Or assume ID order is chronological
        result = orderBy(result, ['id'], ['desc']);
      } else {
        // Top Stories = Sort by Likes
        result = orderBy(result, ['likes'], ['desc']);
      }

      return result;
    });

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value;
    };

    const selectSort = (option) => {
      sortOption.value = option;
      sortOpen.value = false;
      signatureStore.news_feed_filters_applied = true; // FSM Effect
    };

    const toggleFriendsOnly = () => {
      filters.value.friendsOnly = !filters.value.friendsOnly;
      signatureStore.news_feed_filters_applied = true; // FSM Effect
    };

    const applyFilters = () => {
      signatureStore.news_feed_filters_applied = true; // FSM Effect
    };

    const openPost = async (post) => {
      signatureStore.selected_post_id = post.id;
      // Clear viewport anchor (FSM Effect)
      signatureStore.news_feed_viewport_anchor_id = null;
      // Clear filters applied flag (FSM Effect for filtered click)
      if (isFiltered.value) {
        signatureStore.news_feed_filters_applied = null;
      }
      
      await router.push({ name: 'POST_DETAIL', params: { id: post.id } });
    };

    const goToCreatePost = async () => {
      signatureStore.currentPageId = 'CREATE_POST';
      await router.push({ name: 'CREATE_POST' });
    };

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    return {
      sortOpen,
      sortOption,
      filters,
      isFiltered,
      filteredPosts,
      toggleSort,
      selectSort,
      toggleFriendsOnly,
      applyFilters,
      openPost,
      goToCreatePost,
      goHome
    };
  }
}
</script>