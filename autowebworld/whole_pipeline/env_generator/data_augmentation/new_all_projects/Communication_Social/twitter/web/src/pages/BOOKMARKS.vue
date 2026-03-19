<template>
  <div class="flex flex-col min-h-screen bg-black text-white pb-20 sm:pb-0">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md px-4 py-3 flex items-center justify-between border-b border-[#2F3336]">
      <div class="flex items-center gap-4">
        <div id="bookmarks-back-home" @click="handleBackHome" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors sm:hidden">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M7.414 13l5.043 5.04-1.414 1.42L3.586 12l7.457-7.46 1.414 1.42L7.414 11H21v2H7.414z"></path></g></svg>
        </div>
        <div class="flex flex-col">
           <h2 class="text-xl font-bold">Bookmarks</h2>
           <div class="text-sm text-[#71767B]">@myself</div>
        </div>
      </div>
      <div class="p-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
         <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M10.54 1.75h2.92l1.57 2.36c.43.65 1.12 1.07 1.9.98l2.8-.29.77 2.73c.2.73.77 1.3 1.51 1.51l2.73.77-.29 2.8c-.09.78.33 1.47.98 1.9l2.36 1.57v2.92l-2.36 1.57c-.65.43-1.07 1.12-.98 1.9l.29 2.8-2.73.77c-.74.2-1.3.77-1.51 1.51l-.77 2.73-2.8-.29c-.78-.09-1.47.33-1.9.98l-1.57 2.36h-2.92l-1.57-2.36c-.43-.65-1.12-1.07-1.9-.98l-2.8.29-.77-2.73c-.2-.73-.77-1.3-1.51-1.51l-2.73-.77.29-2.8c.09-.78-.33-1.47-.98-1.9l-2.36-1.57v-2.92l2.36-1.57c.65-.43 1.07-1.12.98-1.9l-.29-2.8 2.73-.77c.74-.2 1.3-.77 1.51-1.51l.77-2.73 2.8.29c.78.09 1.47-.33 1.9-.98l1.57-2.36zM12 15.5c1.93 0 3.5-1.57 3.5-3.5s-1.57-3.5-3.5-3.5-3.5 1.57-3.5 3.5 1.57 3.5 3.5 3.5z"></path></g></svg>
      </div>
    </div>

    <!-- Filters -->
    <div class="p-4 border-b border-[#2F3336] flex flex-wrap gap-4 text-sm text-[#71767B]">
        <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
            <input id="bookmarks-filter-media-checkbox" type="checkbox" v-model="filterMedia" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
            Media Only
        </label>

        <!-- Sort -->
        <div class="relative">
            <div id="bookmarks-sort-dropdown" @click="showSortDropdown = !showSortDropdown" class="flex items-center gap-1 cursor-pointer hover:text-white">
                <span>{{ sortOption === 'latest' ? 'Latest' : 'Oldest' }}</span>
                <svg viewBox="0 0 24 24" aria-hidden="true" class="h-4 w-4 fill-current"><g><path d="M3.543 8.96l1.414-1.42L12 14.59l7.043-7.05 1.414 1.42L12 17.41 3.543 8.96z"></path></g></svg>
            </div>
            <div v-if="showSortDropdown" class="absolute top-full left-0 mt-2 bg-black border border-[#2F3336] rounded-lg shadow-xl z-50 py-2 w-32">
                <div id="bookmarks-sort-latest" @click="handleSort('latest')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Latest</div>
                <div id="bookmarks-sort-oldest" @click="handleSort('oldest')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Oldest</div>
            </div>
        </div>
    </div>

    <!-- List -->
    <div id="bookmarks-list-container" class="flex flex-col divide-y divide-[#2F3336]">
       <div id="bookmarks-list">
          <div v-if="filteredBookmarks.length === 0" class="p-8 text-center text-[#71767B]">
              No bookmarks found.
          </div>
          
          <div 
             v-for="tweet in filteredBookmarks" 
             :key="tweet.id" 
             :class="getTweetClass(tweet)"
             class="p-4 hover:bg-white/[0.03] transition-colors cursor-pointer"
             @click="handleOpenTweet(tweet)"
          >
             <div class="flex gap-3">
                <div class="flex-shrink-0 w-12 h-12 rounded-full overflow-hidden bg-gray-700">
                    <img :src="getAuthor(tweet.author_id)?.avatar || '/images/photo1766328910.jpg'" alt="avatar" class="w-full h-full object-cover">
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
  name: 'BOOKMARKS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const filterMedia = ref(false);
    const sortOption = ref(null);
    const showSortDropdown = ref(false);

    const filteredBookmarks = computed(() => {
        // Map bookmark IDs to tweets
        let result = dataStore.bookmarks.map(b => {
            const t = dataStore.getTweetById(b.tweet_id);
            return t ? { ...t, saved_at: b.saved_at } : null;
        }).filter(t => t !== null);

        if (filterMedia.value) {
            result = result.filter(t => t.has_media);
        }

        if (sortOption.value === 'oldest') {
            result.sort((a, b) => new Date(a.saved_at) - new Date(b.saved_at));
        } else if (sortOption.value === 'latest') {
             result.sort((a, b) => new Date(b.saved_at) - new Date(a.saved_at));
        }

        return result;
    });

    const getAuthor = (id) => dataStore.getUserById(id);

    const getTweetClass = (tweet) => {
        const classes = [`data-id-${tweet.id}`];
        if (filterMedia.value || sortOption.value) classes.push('tweet-filtered');
        else classes.push('tweet-visible');
        return classes.join(' ');
    };

    const handleSort = (opt) => {
        sortOption.value = opt;
        signatureStore.bookmarks_filters_applied = true;
        showSortDropdown.value = false;
    };

    const handleOpenTweet = (tweet) => {
        signatureStore.selected_tweet_id = tweet.id;
        signatureStore.setCurrentPageId('TWEET_DETAIL');
        signatureStore.bookmarks_filters_applied = null;
        router.push({ name: 'TWEET_DETAIL', params: { tweet_id: tweet.id } });
    };

    const handleBackHome = () => {
        signatureStore.setCurrentPageId('HOME');
        router.push({ name: 'HOME' });
    };
    
    watch([filterMedia, sortOption], () => {
       signatureStore.bookmarks_filters_applied = true;
    });

    return {
        filterMedia,
        sortOption,
        showSortDropdown,
        filteredBookmarks,
        getAuthor,
        getTweetClass,
        handleSort,
        handleOpenTweet,
        handleBackHome
    };
  }
}
</script>