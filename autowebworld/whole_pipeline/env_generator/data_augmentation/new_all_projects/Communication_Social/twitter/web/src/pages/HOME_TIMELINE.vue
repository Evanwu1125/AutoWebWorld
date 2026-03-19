<template>
  <div class="flex flex-col min-h-screen bg-black text-white pb-20 sm:pb-0">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md border-b border-[#2F3336] px-4 py-3 cursor-pointer"
         @click="scrollToTop">
      <h2 class="text-xl font-bold">Home</h2>
    </div>

    <!-- Compose Tweet Teaser (Desktop) -->
    <div class="hidden sm:block border-b border-[#2F3336] px-4 py-3">
      <div class="flex gap-4">
        <div class="w-12 h-12 rounded-full overflow-hidden bg-gray-700">
           <img :src="currentUser?.avatar || '/images/photo1766328968.jpg'" alt="avatar" class="w-full h-full object-cover">
        </div>
        <div class="flex-1 cursor-text" @click="handleComposeTweet">
           <div class="text-[#71767B] text-xl py-3">What is happening?!</div>
           <div class="flex justify-between items-center mt-2 pt-2 border-t border-[#2F3336] border-opacity-0">
             <div class="flex gap-2 text-[#1D9BF0]">
               <!-- Icons -->
             </div>
             <button id="primary-compose-tweet" class="bg-[#1D9BF0] text-white font-bold rounded-full px-4 py-1.5 hover:bg-[#1A8CD8] transition-colors">
               Post
             </button>
           </div>
        </div>
      </div>
    </div>

    <!-- Filters & Search -->
    <div class="border-b border-[#2F3336] p-4 flex flex-col gap-4">
      <!-- Search -->
      <div class="relative group w-full">
         <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 text-gray-500"><g><path d="M10.25 3.75c-3.59 0-6.5 2.91-6.5 6.5s2.91 6.5 6.5 6.5c1.795 0 3.419-.726 4.596-1.904 1.178-1.177 1.904-2.801 1.904-4.596 0-3.59-2.91-6.5-6.5-6.5zm-8.5 6.5c0-4.694 3.806-8.5 8.5-8.5s8.5 3.806 8.5 8.5c0 1.986-.73 3.815-1.945 5.232l4.944 4.942-1.414 1.415-4.942-4.944C14.065 18.02 12.236 18.75 10.25 18.75c-4.694 0-8.5-3.806-8.5-8.5z"></path></g></svg>
         </div>
         <input 
            id="timeline-search-input"
            v-model="searchQuery" 
            @keydown.enter="handleSearch"
            type="text" 
            placeholder="Search timeline" 
            class="w-full bg-[#202327] text-white rounded-full py-2 pl-10 pr-4 focus:outline-none focus:ring-1 focus:ring-[#1D9BF0] border border-transparent placeholder-gray-500 text-sm"
         >
      </div>

      <!-- Filter Controls Row -->
      <div class="flex flex-wrap items-center gap-4 text-sm text-[#71767B]">
        <!-- Checkboxes -->
        <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
          <input id="timeline-filter-following-checkbox" type="checkbox" v-model="filterFollowing" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
          Following
        </label>
        
        <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
          <input id="timeline-filter-media-checkbox" type="checkbox" v-model="filterMediaOnly" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
          Media
        </label>

        <!-- Sort Dropdown -->
        <div class="relative">
          <div id="timeline-sort-dropdown" @click="toggleSortDropdown" class="flex items-center gap-1 cursor-pointer hover:text-white">
             <span>{{ sortOption === 'latest' ? 'Latest' : 'Top' }}</span>
             <svg viewBox="0 0 24 24" aria-hidden="true" class="h-4 w-4 fill-current"><g><path d="M3.543 8.96l1.414-1.42L12 14.59l7.043-7.05 1.414 1.42L12 17.41 3.543 8.96z"></path></g></svg>
          </div>
          <div v-if="showSortDropdown" class="absolute top-full left-0 mt-2 bg-black border border-[#2F3336] rounded-lg shadow-xl z-50 py-2 w-32">
             <div id="timeline-sort-latest" @click="handleSort('latest')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Latest</div>
             <div id="timeline-sort-top" @click="handleSort('top')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Top</div>
          </div>
        </div>
      </div>

      <!-- Slider -->
      <div class="flex flex-col gap-1 w-full max-w-xs">
         <div class="flex justify-between text-xs text-[#71767B]">
            <span>Min Likes: {{ filterLikes }}</span>
            <span>{{ maxLikes }}</span>
         </div>
         <input 
            id="timeline-filter-engagement-slider" 
            type="range" 
            v-model.number="filterLikes" 
            :min="0" 
            :max="maxLikes" 
            step="100"
            class="w-full h-1 bg-[#2F3336] rounded-lg appearance-none cursor-pointer accent-[#1D9BF0]"
         >
      </div>
    </div>

    <!-- Tweets List -->
    <div id="timeline-list-container" class="flex flex-col divide-y divide-[#2F3336]">
       <div v-if="filteredTweets.length === 0" class="p-8 text-center text-[#71767B]">
          No tweets found.
       </div>

       <div 
          v-for="tweet in filteredTweets" 
          :key="tweet.id" 
          :class="getTweetClass(tweet)"
          class="p-4 hover:bg-white/[0.03] transition-colors cursor-pointer"
          @click="handleOpenTweet(tweet)"
       >
         <div class="flex gap-3">
            <div class="flex-shrink-0 w-12 h-12 rounded-full overflow-hidden bg-gray-700">
               <img :src="getAuthor(tweet.author_id)?.avatar || '/images/photo1766328968.jpg'" alt="avatar" class="w-full h-full object-cover">
            </div>
            <div class="flex-1 min-w-0">
               <div class="flex items-center gap-1 text-[#71767B] text-sm truncate">
                  <span class="font-bold text-white truncate">{{ getAuthor(tweet.author_id)?.name }}</span>
                  <span class="truncate">{{ getAuthor(tweet.author_id)?.handle }}</span>
                  <span>·</span>
                  <span>{{ tweet.timestamp }}</span>
               </div>
               
               <div class="mt-1 text-white whitespace-pre-wrap break-words text-[15px] leading-6">
                  {{ tweet.content }}
               </div>

               <div v-if="tweet.has_media && tweet.media_url" class="mt-3 rounded-2xl overflow-hidden border border-[#2F3336]">
                  <img :src="tweet.media_url" alt="media" class="w-full max-h-[500px] object-cover">
               </div>

               <!-- Action Buttons -->
               <div class="flex justify-between mt-3 max-w-md text-[#71767B] text-sm">
                  <div class="group flex items-center gap-2 hover:text-[#1D9BF0]">
                     <div class="p-2 rounded-full group-hover:bg-[#1D9BF0]/10 transition-colors">
                        <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M1.751 10c0-4.42 3.584-8 8.005-8h4.366c4.49 0 8.129 3.64 8.129 8.13 0 2.96-1.607 5.68-4.196 7.11l-8.054 4.46v-3.69h-.067c-4.49.1-8.183-3.51-8.183-8.01zm8.005-6c-3.317 0-6.005 2.69-6.005 6 0 3.37 2.77 6.08 6.138 6.01l.351-.01h1.761v2.3l5.087-2.81c1.951-1.08 3.163-3.13 3.163-5.36 0-3.39-2.744-6.13-6.129-6.13H9.756z"></path></g></svg>
                     </div>
                     <span>{{ tweet.replies }}</span>
                  </div>
                  <div class="group flex items-center gap-2 hover:text-[#00BA7C]">
                     <div class="p-2 rounded-full group-hover:bg-[#00BA7C]/10 transition-colors">
                        <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M4.5 3.88l4.432 4.14-1.364 1.46L5.5 7.55V16c0 1.1.896 2 2 2H13v2H7.5c-2.209 0-4-1.79-4-4V7.55L1.432 9.48.068 8.02 4.5 3.88zM16.5 6H11V4h5.5c2.209 0 4 1.79 4 4v8.45l2.068-1.93 1.364 1.46-4.432 4.14-4.432-4.14 1.364-1.46 2.068 1.93V8c0-1.1-.896-2-2-2z"></path></g></svg>
                     </div>
                     <span>{{ tweet.retweets }}</span>
                  </div>
                  <div class="group flex items-center gap-2 hover:text-[#F91880]">
                     <div class="p-2 rounded-full group-hover:bg-[#F91880]/10 transition-colors">
                        <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M16.697 5.5c-1.222-.06-2.679.51-3.89 2.16l-.805 1.09-.806-1.09C9.984 6.01 8.526 5.44 7.304 5.5c-1.243.07-2.349.78-2.91 1.91-.552 1.12-.633 2.78.479 4.82 1.074 1.97 3.257 4.27 7.129 6.61 3.87-2.34 6.052-4.64 7.126-6.61 1.111-2.04 1.03-3.7.477-4.82-.561-1.13-1.666-1.84-2.908-1.91zm4.187 7.69c-1.351 2.48-4.001 5.12-8.379 7.67l-.503.3-.504-.3c-4.379-2.55-7.029-5.19-8.382-7.67-1.36-2.5-1.41-4.86-.514-6.67.887-1.79 2.647-2.91 4.601-3.01 1.651-.09 3.368.56 4.798 2.01 1.429-1.45 3.146-2.1 4.796-2.01 1.954.1 3.714 1.22 4.605 3.01.894 1.81.846 4.17-.514 6.67z"></path></g></svg>
                     </div>
                     <span>{{ tweet.likes }}</span>
                  </div>
               </div>
            </div>
         </div>
       </div>
    </div>
    
    <!-- Footer Back Link -->
    <div class="p-4 border-t border-[#2F3336] mt-4 flex justify-center">
       <button id="back-to-root-home" @click="handleBackToRoot" class="text-[#1D9BF0] hover:underline font-bold">
          Back to Start
       </button>
    </div>
  </div>
</template>

<script>
import { computed, ref, onMounted, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';
import _ from 'lodash-es';

export default {
  name: 'HOME_TIMELINE',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    // State for filters
    const filterFollowing = ref(false);
    const filterMediaOnly = ref(false);
    const filterLikes = ref(0);
    const sortOption = ref(null); // 'latest', 'top' or null initially
    const showSortDropdown = ref(false);
    const searchQuery = ref('');

    // Computed Properties
    const currentUser = computed(() => dataStore.getUserById('user_me'));
    const maxLikes = computed(() => {
       if (!dataStore.tweets.length) return 10000;
       return Math.max(...dataStore.tweets.map(t => t.likes)) || 10000;
    });

    const filteredTweets = computed(() => {
       let result = [...dataStore.tweets];

       // Search
       if (signatureStore.matched_tweet_id) {
          return result.filter(t => t.id === signatureStore.matched_tweet_id);
       }
       if (searchQuery.value) {
          // handled below in handleSearch for FSM state update, but for immediate filtering:
          // In this simplified Vue model, we can filter here directly for visual feedback
       }

       // Filters
       if (filterFollowing.value) {
          // Mock following logic: Assume we follow everyone except 'user_me' for demo purposes or use specific logic
          // Here, let's say we follow ids ['u1', 'u2', 'u3', 'u4', 'u5']
          const followingIds = ['u1', 'u2', 'u3', 'u4', 'u5']; 
          result = result.filter(t => followingIds.includes(t.author_id));
       }

       if (filterMediaOnly.value) {
          result = result.filter(t => t.has_media);
       }

       if (filterLikes.value > 0) {
          result = result.filter(t => t.likes >= filterLikes.value);
       }

       // Sorting
       if (sortOption.value === 'latest') {
          // Rough timestamp sort (mock data has '2h', '1d', etc. - parsing is hard, let's rely on array order for now or mock timestamp values)
          // Ideally mock data has real ISO dates. Assuming array order is somewhat chronological or random.
          // Let's just reverse for "Latest" if default is old
       } else if (sortOption.value === 'top') {
          result.sort((a, b) => b.likes - a.likes);
       }

       return result;
    });

    // Helper Functions
    const getAuthor = (id) => dataStore.getUserById(id);

    const getTweetClass = (tweet) => {
       const classes = [`data-id-${tweet.id}`];
       // Determine if it's a search result, filtered result, or visible
       if (signatureStore.matched_tweet_id === tweet.id) {
          classes.push('tweet-search-result');
       } else if (filterFollowing.value || filterMediaOnly.value || filterLikes.value > 0 || sortOption.value) {
          classes.push('tweet-filtered');
       } else {
          classes.push('tweet-visible');
       }
       return classes.join(' ');
    };

    // Actions
    const handleComposeTweet = () => {
       signatureStore.setCurrentPageId('COMPOSE_TWEET');
       router.push({ name: 'COMPOSE_TWEET' });
    };

    const handleSearch = () => {
       if (!searchQuery.value) return;
       
       // Find a tweet matching content
       const match = dataStore.tweets.find(t => t.content.toLowerCase().includes(searchQuery.value.toLowerCase()));
       if (match) {
          signatureStore.matched_tweet_id = match.id;
          signatureStore.home_timeline_has_searched = true;
          // UI will reactively update via filteredTweets computed
       }
    };

    const toggleSortDropdown = () => {
       showSortDropdown.value = !showSortDropdown.value;
    };

    const handleSort = (option) => {
       sortOption.value = option;
       signatureStore.home_timeline_filters_applied = true;
       showSortDropdown.value = false;
    };

    const handleOpenTweet = (tweet) => {
       signatureStore.selected_tweet_id = tweet.id;
       signatureStore.setCurrentPageId('TWEET_DETAIL');
       // Clear filters/search flags based on FSM effects if needed, 
       // but router beforeEach handles most page id updates.
       // FSM Effects: clear home_timeline_filters_applied, clear matched_tweet_id etc.
       signatureStore.home_timeline_filters_applied = null;
       signatureStore.matched_tweet_id = null;
       signatureStore.home_timeline_has_searched = null;

       router.push({ name: 'TWEET_DETAIL', params: { tweet_id: tweet.id } });
    };

    const handleBackToRoot = () => {
       signatureStore.setCurrentPageId('HOME');
       router.push({ name: 'HOME' });
    };

    const scrollToTop = () => {
       window.scrollTo({ top: 0, behavior: 'smooth' });
    };
    
    // Watchers
    watch([filterFollowing, filterMediaOnly, filterLikes, sortOption], () => {
       signatureStore.home_timeline_filters_applied = true;
    });

    return {
       currentUser,
       searchQuery,
       filterFollowing,
       filterMediaOnly,
       filterLikes,
       maxLikes,
       sortOption,
       showSortDropdown,
       filteredTweets,
       getAuthor,
       getTweetClass,
       handleComposeTweet,
       handleSearch,
       toggleSortDropdown,
       handleSort,
       handleOpenTweet,
       handleBackToRoot,
       scrollToTop
    };
  }
}
</script>