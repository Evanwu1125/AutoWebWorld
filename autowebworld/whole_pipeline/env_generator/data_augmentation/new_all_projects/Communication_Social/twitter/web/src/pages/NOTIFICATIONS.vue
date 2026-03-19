<template>
  <div class="flex flex-col min-h-screen bg-black text-white pb-20 sm:pb-0">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md px-4 py-3 flex items-center justify-between border-b border-[#2F3336]">
      <div class="flex items-center gap-4">
        <div id="notifications-back-home" @click="handleBackHome" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors sm:hidden">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M7.414 13l5.043 5.04-1.414 1.42L3.586 12l7.457-7.46 1.414 1.42L7.414 11H21v2H7.414z"></path></g></svg>
        </div>
        <h2 class="text-xl font-bold">Notifications</h2>
      </div>
      <div class="p-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
         <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M10.54 1.75h2.92l1.57 2.36c.43.65 1.12 1.07 1.9.98l2.8-.29.77 2.73c.2.73.77 1.3 1.51 1.51l2.73.77-.29 2.8c-.09.78.33 1.47.98 1.9l2.36 1.57v2.92l-2.36 1.57c-.65.43-1.07 1.12-.98 1.9l.29 2.8-2.73.77c-.74.2-1.3.77-1.51 1.51l-.77 2.73-2.8-.29c-.78-.09-1.47.33-1.9.98l-1.57 2.36h-2.92l-1.57-2.36c-.43-.65-1.12-1.07-1.9-.98l-2.8.29-.77-2.73c-.2-.73-.77-1.3-1.51-1.51l-2.73-.77.29-2.8c.09-.78-.33-1.47-.98-1.9l-2.36-1.57v-2.92l2.36-1.57c.65-.43 1.07-1.12.98-1.9l-.29-2.8 2.73-.77c.74-.2 1.3-.77 1.51-1.51l.77-2.73 2.8.29c.78.09 1.47-.33 1.9-.98l1.57-2.36zM12 15.5c1.93 0 3.5-1.57 3.5-3.5s-1.57-3.5-3.5-3.5-3.5 1.57-3.5 3.5 1.57 3.5 3.5 3.5z"></path></g></svg>
      </div>
    </div>

    <!-- Filters -->
    <div class="p-4 border-b border-[#2F3336] flex flex-wrap gap-4 text-sm text-[#71767B]">
        <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
           <input id="notifications-filter-mentions-checkbox" type="checkbox" v-model="filterMentions" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
           Mentions
        </label>
        <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
           <input id="notifications-filter-follows-checkbox" type="checkbox" v-model="filterFollows" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
           Follows
        </label>

        <!-- Sort -->
         <div class="relative">
            <div id="notifications-sort-dropdown" @click="showSortDropdown = !showSortDropdown" class="flex items-center gap-1 cursor-pointer hover:text-white">
                <span>{{ sortOption === 'unread' ? 'Unread First' : 'Latest' }}</span>
                <svg viewBox="0 0 24 24" aria-hidden="true" class="h-4 w-4 fill-current"><g><path d="M3.543 8.96l1.414-1.42L12 14.59l7.043-7.05 1.414 1.42L12 17.41 3.543 8.96z"></path></g></svg>
            </div>
            <div v-if="showSortDropdown" class="absolute top-full left-0 mt-2 bg-black border border-[#2F3336] rounded-lg shadow-xl z-50 py-2 w-36">
                <div id="notifications-sort-unread" @click="handleSort('unread')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Unread First</div>
                <div id="notifications-sort-latest" @click="handleSort('latest')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Latest</div>
            </div>
         </div>
    </div>

    <!-- List -->
    <div id="notifications-list" class="flex flex-col divide-y divide-[#2F3336]">
       <div v-if="filteredNotifications.length === 0" class="p-8 text-center text-[#71767B]">
           No notifications.
       </div>

       <div 
          v-for="notif in filteredNotifications" 
          :key="notif.id"
          class="notification-item p-4 hover:bg-white/[0.03] transition-colors cursor-pointer flex gap-3"
          :class="!notif.is_read ? 'bg-[#16181C]' : ''"
          @click="handleOpenItem(notif)"
       >
          <!-- Icon based on type -->
          <div class="w-8 flex-shrink-0 flex justify-end">
             <svg v-if="notif.type === 'like'" viewBox="0 0 24 24" class="h-7 w-7 fill-[#F91880]"><g><path d="M16.697 5.5c-1.222-.06-2.679.51-3.89 2.16l-.805 1.09-.806-1.09C9.984 6.01 8.526 5.44 7.304 5.5c-1.243.07-2.349.78-2.91 1.91-.552 1.12-.633 2.78.479 4.82 1.074 1.97 3.257 4.27 7.129 6.61 3.87-2.34 6.052-4.64 7.126-6.61 1.111-2.04 1.03-3.7.477-4.82-.561-1.13-1.666-1.84-2.908-1.91zm4.187 7.69c-1.351 2.48-4.001 5.12-8.379 7.67l-.503.3-.504-.3c-4.379-2.55-7.029-5.19-8.382-7.67-1.36-2.5-1.41-4.86-.514-6.67.887-1.79 2.647-2.91 4.601-3.01 1.651-.09 3.368.56 4.798 2.01 1.429-1.45 3.146-2.1 4.796-2.01 1.954.1 3.714 1.22 4.605 3.01.894 1.81.846 4.17-.514 6.67z"></path></g></svg>
             <svg v-else-if="notif.type === 'retweet'" viewBox="0 0 24 24" class="h-7 w-7 fill-[#00BA7C]"><g><path d="M4.5 3.88l4.432 4.14-1.364 1.46L5.5 7.55V16c0 1.1.896 2 2 2H13v2H7.5c-2.209 0-4-1.79-4-4V7.55L1.432 9.48.068 8.02 4.5 3.88zM16.5 6H11V4h5.5c2.209 0 4 1.79 4 4v8.45l2.068-1.93 1.364 1.46-4.432 4.14-4.432-4.14 1.364-1.46 2.068 1.93V8c0-1.1-.896-2-2-2z"></path></g></svg>
             <svg v-else-if="notif.type === 'follow'" viewBox="0 0 24 24" class="h-7 w-7 fill-[#1D9BF0]"><g><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm5 11h-4v4h-2v-4H7v-2h4V7h2v4h4v2z"></path></g></svg>
             <svg v-else-if="notif.type === 'mention' || notif.type === 'reply'" viewBox="0 0 24 24" class="h-7 w-7 fill-[#1D9BF0]"><g><path d="M12 1.75a8.25 8.25 0 00-8.25 8.25v2.887c0 2.225-1.077 4.192-2.81 5.378-.458.314-.492.969-.074 1.334.814.71 1.868 1.15 3.033 1.15H10.5a3.501 3.501 0 006.999 0h6.6c1.165 0 2.22-.44 3.034-1.15.418-.365.385-1.02-.073-1.334-1.734-1.186-2.811-3.153-2.811-5.378V10A8.25 8.25 0 0012 1.75zM14 22a1.5 1.5 0 11-3 0h3z"></path></g></svg>
          </div>

          <div class="flex-1 flex flex-col gap-2">
             <div class="w-8 h-8 rounded-full overflow-hidden bg-gray-700">
                <img :src="getUser(notif.user_id)?.avatar || '/images/photo1766328831.jpg'" alt="avatar" class="w-full h-full object-cover">
             </div>
             
             <div class="text-[15px]">
                <span class="font-bold">{{ getUser(notif.user_id)?.name }}</span>
                <span v-if="notif.type === 'like'"> liked your post</span>
                <span v-if="notif.type === 'retweet'"> reposted your post</span>
                <span v-if="notif.type === 'follow'"> followed you</span>
                <span v-if="notif.type === 'mention'"> mentioned you</span>
                <span v-if="notif.type === 'reply'"> replied to your post</span>
             </div>

             <div v-if="notif.text" class="text-[#71767B]">
                {{ notif.text }}
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
  name: 'NOTIFICATIONS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const filterMentions = ref(false);
    const filterFollows = ref(false);
    const sortOption = ref(null);
    const showSortDropdown = ref(false);

    const filteredNotifications = computed(() => {
        let result = [...dataStore.notifications];

        if (filterMentions.value) {
            result = result.filter(n => n.type === 'mention' || n.type === 'reply');
        }
        if (filterFollows.value) {
            result = result.filter(n => n.type === 'follow');
        }

        if (sortOption.value === 'unread') {
            result.sort((a, b) => (b.is_read === a.is_read) ? 0 : b.is_read ? -1 : 1); // Unread (false) first? Wait, is_read=false comes before true
        }

        return result;
    });

    const getUser = (id) => dataStore.getUserById(id);

    const handleSort = (opt) => {
        sortOption.value = opt;
        signatureStore.notifications_filters_applied = true;
        showSortDropdown.value = false;
    };

    const handleOpenItem = (notif) => {
        // FSM: ACT_NOTIFICATIONS_OPEN_ITEM -> TWEET_DETAIL
        // Only if it has a tweet_id. If it's a follow, maybe it should go to profile?
        // But FSM defines one action: ACT_NOTIFICATIONS_OPEN_ITEM -> TWEET_DETAIL
        // With parameter tweet_id.
        // If tweet_id is null (e.g. follow), this might fail or we should handle gracefully.
        // We'll proceed if tweet_id exists.
        
        if (notif.tweet_id) {
           signatureStore.selected_tweet_id = notif.tweet_id;
           signatureStore.setCurrentPageId('TWEET_DETAIL');
           router.push({ name: 'TWEET_DETAIL', params: { tweet_id: notif.tweet_id } });
        } else if (notif.type === 'follow') {
           // Fallback/Enhancement: Go to user profile?
           // FSM doesn't specify, but for usability:
           signatureStore.user_id = notif.user_id;
           signatureStore.setCurrentPageId('USER_PROFILE_OVERVIEW');
           router.push({ name: 'USER_PROFILE_OVERVIEW', params: { user_id: notif.user_id } });
        }
    };

    const handleBackHome = () => {
        signatureStore.setCurrentPageId('HOME');
        router.push({ name: 'HOME' });
    };
    
    watch([filterMentions, filterFollows, sortOption], () => {
       signatureStore.notifications_filters_applied = true;
    });

    return {
        filterMentions,
        filterFollows,
        sortOption,
        showSortDropdown,
        filteredNotifications,
        getUser,
        handleSort,
        handleOpenItem,
        handleBackHome
    };
  }
}
</script>