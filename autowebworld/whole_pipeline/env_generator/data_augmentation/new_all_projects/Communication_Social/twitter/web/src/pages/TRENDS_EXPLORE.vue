<template>
  <div class="flex flex-col min-h-screen bg-black text-white pb-20 sm:pb-0">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md px-4 py-3 flex items-center gap-4 border-b border-[#2F3336]">
      <div id="trends-back-home" @click="handleBackHome" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors sm:hidden">
         <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M7.414 13l5.043 5.04-1.414 1.42L3.586 12l7.457-7.46 1.414 1.42L7.414 11H21v2H7.414z"></path></g></svg>
      </div>
      <div class="flex-1 relative group">
          <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 text-gray-500"><g><path d="M10.25 3.75c-3.59 0-6.5 2.91-6.5 6.5s2.91 6.5 6.5 6.5c1.795 0 3.419-.726 4.596-1.904 1.178-1.177 1.904-2.801 1.904-4.596 0-3.59-2.91-6.5-6.5-6.5zm-8.5 6.5c0-4.694 3.806-8.5 8.5-8.5s8.5 3.806 8.5 8.5c0 1.986-.73 3.815-1.945 5.232l4.944 4.942-1.414 1.415-4.942-4.944C14.065 18.02 12.236 18.75 10.25 18.75c-4.694 0-8.5-3.806-8.5-8.5z"></path></g></svg>
          </div>
          <input type="text" placeholder="Search Twitter" class="w-full bg-[#202327] text-white rounded-full py-2 pl-10 pr-4 focus:outline-none focus:ring-1 focus:ring-[#1D9BF0] border border-transparent placeholder-gray-500 text-sm">
      </div>
      <div class="p-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
         <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M10.54 1.75h2.92l1.57 2.36c.43.65 1.12 1.07 1.9.98l2.8-.29.77 2.73c.2.73.77 1.3 1.51 1.51l2.73.77-.29 2.8c-.09.78.33 1.47.98 1.9l2.36 1.57v2.92l-2.36 1.57c-.65.43-1.07 1.12-.98 1.9l.29 2.8-2.73.77c-.74.2-1.3.77-1.51 1.51l-.77 2.73-2.8-.29c-.78-.09-1.47.33-1.9.98l-1.57 2.36h-2.92l-1.57-2.36c-.43-.65-1.12-1.07-1.9-.98l-2.8.29-.77-2.73c-.2-.73-.77-1.3-1.51-1.51l-2.73-.77.29-2.8c.09-.78-.33-1.47-.98-1.9l-2.36-1.57v-2.92l2.36-1.57c.65-.43 1.07-1.12.98-1.9l-.29-2.8 2.73-.77c.74-.2 1.3-.77 1.51-1.51l.77-2.73 2.8.29c.78.09 1.47-.33 1.9-.98l1.57-2.36zM12 15.5c1.93 0 3.5-1.57 3.5-3.5s-1.57-3.5-3.5-3.5-3.5 1.57-3.5 3.5 1.57 3.5 3.5 3.5z"></path></g></svg>
      </div>
    </div>

    <!-- Filters -->
    <div class="p-4 border-b border-[#2F3336] flex flex-wrap gap-4 text-sm text-[#71767B]">
        <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
            <input id="trends-filter-nearby-checkbox" type="checkbox" v-model="filterNearby" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
            Nearby
        </label>
        <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
            <input id="trends-filter-sports-checkbox" type="checkbox" v-model="filterSports" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
            Sports
        </label>

        <!-- Sort -->
        <div class="relative">
            <div id="trends-sort-dropdown" @click="showSortDropdown = !showSortDropdown" class="flex items-center gap-1 cursor-pointer hover:text-white">
                <span>{{ sortOption === 'trending' ? 'Trending' : 'Latest' }}</span>
                <svg viewBox="0 0 24 24" aria-hidden="true" class="h-4 w-4 fill-current"><g><path d="M3.543 8.96l1.414-1.42L12 14.59l7.043-7.05 1.414 1.42L12 17.41 3.543 8.96z"></path></g></svg>
            </div>
            <div v-if="showSortDropdown" class="absolute top-full left-0 mt-2 bg-black border border-[#2F3336] rounded-lg shadow-xl z-50 py-2 w-36">
                <div id="trends-sort-trending" @click="handleSort('trending')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Trending</div>
                <div id="trends-sort-latest" @click="handleSort('latest')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Latest</div>
            </div>
        </div>
    </div>

    <!-- Trends List -->
    <div id="trends-list" class="flex flex-col">
       <div v-if="filteredTrends.length === 0" class="p-8 text-center text-[#71767B]">
           No trends found.
       </div>

       <div 
          v-for="(trend, index) in filteredTrends" 
          :key="trend.id"
          class="trend-item p-4 hover:bg-white/[0.03] transition-colors cursor-pointer flex justify-between items-start"
          @click="handleOpenTopic(trend)"
       >
          <div class="flex flex-col">
             <div class="text-sm text-[#71767B] flex items-center gap-1">
                 <span>{{ index + 1 }} · {{ trend.category }} · Trending</span>
             </div>
             <div class="font-bold text-[15px] mt-0.5">{{ trend.name }}</div>
             <div class="text-sm text-[#71767B] mt-0.5">{{ trend.tweets_count }} Tweets</div>
          </div>
          <div class="p-2 -mr-2 rounded-full hover:bg-[#1D9BF0]/10 hover:text-[#1D9BF0] text-[#71767B] cursor-pointer">
             <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M3 12c0-1.1.9-2 2-2s2 .9 2 2-.9 2-2 2-2-.9-2-2zm9 2c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm7 0c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2z"></path></g></svg>
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
  name: 'TRENDS_EXPLORE',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const filterNearby = ref(false);
    const filterSports = ref(false);
    const sortOption = ref(null);
    const showSortDropdown = ref(false);

    const filteredTrends = computed(() => {
        let result = [...dataStore.trends];

        // Mock Nearby logic: shuffle or limit? Let's just shuffle to show change
        if (filterNearby.value) {
            // result = result.slice(0, 5); // Just a subset
        }
        
        if (filterSports.value) {
            result = result.filter(t => t.category === 'Sports');
        }

        if (sortOption.value === 'trending') {
            result.sort((a, b) => (b.is_trending === a.is_trending) ? 0 : b.is_trending ? 1 : -1);
        }

        return result;
    });

    const handleSort = (opt) => {
        sortOption.value = opt;
        signatureStore.trends_filters_applied = true;
        showSortDropdown.value = false;
    };

    const handleOpenTopic = (trend) => {
        signatureStore.topic_id = trend.id; // Store explicitly if needed
        signatureStore.setCurrentPageId('TOPIC_TWEET_LIST');
        router.push({ name: 'TOPIC_TWEET_LIST', params: { topic_id: trend.id } });
    };

    const handleBackHome = () => {
        signatureStore.setCurrentPageId('HOME');
        router.push({ name: 'HOME' });
    };
    
    watch([filterNearby, filterSports, sortOption], () => {
       signatureStore.trends_filters_applied = true;
    });

    return {
        filterNearby,
        filterSports,
        sortOption,
        showSortDropdown,
        filteredTrends,
        handleSort,
        handleOpenTopic,
        handleBackHome
    };
  }
}
</script>