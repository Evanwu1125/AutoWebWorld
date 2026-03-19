<template>
  <div class="flex flex-col min-h-screen bg-black text-white pb-20 sm:pb-0">
     <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md px-4 py-3 flex items-center gap-4 border-b border-[#2F3336]">
      <div id="profile-tweets-back-overview" @click="handleBack" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
         <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M7.414 13l5.043 5.04-1.414 1.42L3.586 12l7.457-7.46 1.414 1.42L7.414 11H21v2H7.414z"></path></g></svg>
      </div>
      <div>
         <h2 class="text-xl font-bold">Posts</h2>
      </div>
    </div>

    <!-- Filters -->
    <div class="p-4 border-b border-[#2F3336] flex flex-col gap-4">
        <!-- Search -->
        <div class="relative group w-full">
            <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 text-gray-500"><g><path d="M10.25 3.75c-3.59 0-6.5 2.91-6.5 6.5s2.91 6.5 6.5 6.5c1.795 0 3.419-.726 4.596-1.904 1.178-1.177 1.904-2.801 1.904-4.596 0-3.59-2.91-6.5-6.5-6.5zm-8.5 6.5c0-4.694 3.806-8.5 8.5-8.5s8.5 3.806 8.5 8.5c0 1.986-.73 3.815-1.945 5.232l4.944 4.942-1.414 1.415-4.942-4.944C14.065 18.02 12.236 18.75 10.25 18.75c-4.694 0-8.5-3.806-8.5-8.5z"></path></g></svg>
            </div>
            <input 
                id="profile-tweets-search-input"
                v-model="searchQuery" 
                @keydown.enter="handleSearch"
                type="text" 
                placeholder="Search posts" 
                class="w-full bg-[#202327] text-white rounded-full py-2 pl-10 pr-4 focus:outline-none focus:ring-1 focus:ring-[#1D9BF0] border border-transparent placeholder-gray-500 text-sm"
            >
        </div>

        <div class="flex flex-wrap items-center gap-4 text-sm text-[#71767B]">
             <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
                <input id="profile-filter-replies-checkbox" type="checkbox" v-model="filterReplies" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
                Show Replies
             </label>
             <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
                <input id="profile-filter-media-checkbox" type="checkbox" v-model="filterMedia" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
                Media Only
             </label>
             
             <!-- Sort -->
             <div class="relative">
                <div id="profile-tweets-sort-dropdown" @click="showSortDropdown = !showSortDropdown" class="flex items-center gap-1 cursor-pointer hover:text-white">
                    <span>{{ sortOption === 'latest' ? 'Latest' : 'Pinned' }}</span>
                    <svg viewBox="0 0 24 24" aria-hidden="true" class="h-4 w-4 fill-current"><g><path d="M3.543 8.96l1.414-1.42L12 14.59l7.043-7.05 1.414 1.42L12 17.41 3.543 8.96z"></path></g></svg>
                </div>
                <div v-if="showSortDropdown" class="absolute top-full left-0 mt-2 bg-black border border-[#2F3336] rounded-lg shadow-xl z-50 py-2 w-32">
                    <div id="profile-tweets-sort-pinned" @click="handleSort('pinned')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Pinned</div>
                    <div id="profile-tweets-sort-latest" @click="handleSort('latest')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Latest</div>
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
                id="profile-filter-likes-slider" 
                type="range" 
                v-model.number="filterLikes" 
                :min="0" 
                :max="maxLikes" 
                step="10"
                class="w-full h-1 bg-[#2F3336] rounded-lg appearance-none cursor-pointer accent-[#1D9BF0]"
            >
        </div>
    </div>

    <!-- List -->
    <div id="profile-tweets-list-container" class="flex flex-col divide-y divide-[#2F3336]">
       <div id="profile-tweets-list">
          <div v-if="filteredTweets.length === 0" class="p-8 text-center text-[#71767B]">
              No posts found.
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
                    <img :src="getAuthor(tweet.author_id)?.avatar || '/images/photo1766328613.jpg'" alt="avatar" class="w-full h-full object-cover">
                </div>
                <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-1 text-[#71767B] text-sm">
                        <span class="font-bold text-white">{{ getAuthor(tweet.author_id)?.name }}</span>
                        <span>{{ getAuthor(tweet.author_id)?.handle }}</span>
                        <span>·</span>
                        <span>{{ tweet.timestamp }}</span>
                    </div>
                    <div class="mt-1 text-white whitespace-pre-wrap break-words text-[15px] leading-6">{{ tweet.content }}</div>
                    <div v-if="tweet.has_media" class="mt-3 rounded-2xl overflow-hidden border border-[#2F3336]">
                        <img :src="tweet.media_url" alt="media" class="w-full max-h-[400px] object-cover">
                    </div>
                </div>
             </div>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'PROFILE_TWEETS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const searchQuery = ref('');
    const filterReplies = ref(false);
    const filterMedia = ref(false);
    const filterLikes = ref(0);
    const sortOption = ref(null);
    const showSortDropdown = ref(false);

    const userId = computed(() => signatureStore.profile_user_id || 'user_me');

    const maxLikes = computed(() => {
        const userTweets = dataStore.tweets.filter(t => t.author_id === userId.value);
        if (!userTweets.length) return 1000;
        return Math.max(...userTweets.map(t => t.likes)) || 1000;
    });

    const filteredTweets = computed(() => {
       let result = dataStore.tweets.filter(t => t.author_id === userId.value);

       if (signatureStore.matched_tweet_id) {
           return result.filter(t => t.id === signatureStore.matched_tweet_id);
       }
       if (searchQuery.value) {
           // handled via handleSearch generally
       }

       if (!filterReplies.value) {
           // Usually tweets list hides replies unless asked, or maybe it shows only main tweets. 
           // Mock data structure: replies usually have parent_id. 
           // Our mock data is simple. Let's assume all are tweets unless we added 'is_reply'.
           // Let's assume filterReplies=false means show all? Or hide replies?
           // Typically "Show Replies" is a separate tab or toggle.
       }
       
       if (filterMedia.value) {
           result = result.filter(t => t.has_media);
       }
       
       if (filterLikes.value > 0) {
           result = result.filter(t => t.likes >= filterLikes.value);
       }

       if (sortOption.value === 'pinned') {
           // Mock pinned logic (e.g., specific ID or just top)
       } else if (sortOption.value === 'latest') {
           // mock sort
       }

       return result;
    });

    const getAuthor = (id) => dataStore.getUserById(id);

    const getTweetClass = (tweet) => {
        const classes = [`data-id-${tweet.id}`];
        if (signatureStore.matched_tweet_id === tweet.id) classes.push('tweet-search-result');
        else if (filterReplies.value || filterMedia.value || filterLikes.value > 0 || sortOption.value) classes.push('tweet-filtered');
        else classes.push('tweet-visible');
        return classes.join(' ');
    };

    const handleSearch = () => {
        if (!searchQuery.value) return;
        const match = filteredTweets.value.find(t => t.content.toLowerCase().includes(searchQuery.value.toLowerCase()));
        if (match) {
            signatureStore.matched_tweet_id = match.id;
            signatureStore.profile_tweets_has_searched = true;
        }
    };

    const handleSort = (opt) => {
        sortOption.value = opt;
        signatureStore.profile_tweets_filters_applied = true;
        showSortDropdown.value = false;
    };

    const handleOpenTweet = (tweet) => {
        signatureStore.selected_tweet_id = tweet.id;
        signatureStore.setCurrentPageId('TWEET_DETAIL');
        signatureStore.profile_tweets_filters_applied = null;
        signatureStore.matched_tweet_id = null;
        signatureStore.profile_tweets_has_searched = null;
        router.push({ name: 'TWEET_DETAIL', params: { tweet_id: tweet.id } });
    };

    const handleBack = () => {
        signatureStore.setCurrentPageId('PROFILE_OVERVIEW');
        router.push({ name: 'PROFILE_OVERVIEW' });
    };
    
    watch([filterReplies, filterMedia, filterLikes, sortOption], () => {
       signatureStore.profile_tweets_filters_applied = true;
    });

    return {
        searchQuery,
        filterReplies,
        filterMedia,
        filterLikes,
        maxLikes,
        sortOption,
        showSortDropdown,
        filteredTweets,
        getAuthor,
        getTweetClass,
        handleSearch,
        handleSort,
        handleOpenTweet,
        handleBack
    };
  }
}
</script>