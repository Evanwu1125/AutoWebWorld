<template>
  <div class="flex flex-col min-h-screen bg-black text-white pb-20 sm:pb-0">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md px-4 py-3 flex items-center justify-between border-b border-[#2F3336]">
      <div class="flex items-center gap-4">
        <div id="messages-back-home" @click="handleBackHome" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors sm:hidden">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M7.414 13l5.043 5.04-1.414 1.42L3.586 12l7.457-7.46 1.414 1.42L7.414 11H21v2H7.414z"></path></g></svg>
        </div>
        <h2 class="text-xl font-bold">Messages</h2>
      </div>
      <div class="flex gap-2">
         <div class="p-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M10.54 1.75h2.92l1.57 2.36c.43.65 1.12 1.07 1.9.98l2.8-.29.77 2.73c.2.73.77 1.3 1.51 1.51l2.73.77-.29 2.8c-.09.78.33 1.47.98 1.9l2.36 1.57v2.92l-2.36 1.57c-.65.43-1.07 1.12-.98 1.9l.29 2.8-2.73.77c-.74.2-1.3.77-1.51 1.51l-.77 2.73-2.8-.29c-.78-.09-1.47.33-1.9.98l-1.57 2.36h-2.92l-1.57-2.36c-.43-.65-1.12-1.07-1.9-.98l-2.8.29-.77-2.73c-.2-.73-.77-1.3-1.51-1.51l-2.73-.77.29-2.8c.09-.78-.33-1.47-.98-1.9l-2.36-1.57v-2.92l2.36-1.57c.65-.43 1.07-1.12.98-1.9l-.29-2.8 2.73-.77c.74-.2 1.3-.77 1.51-1.51l.77-2.73 2.8.29c.78.09 1.47-.33 1.9-.98l1.57-2.36zM12 15.5c1.93 0 3.5-1.57 3.5-3.5s-1.57-3.5-3.5-3.5-3.5 1.57-3.5 3.5 1.57 3.5 3.5 3.5z"></path></g></svg>
         </div>
         <div id="messages-new-message" @click="handleNewMessage" class="p-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
            <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M1.998 5.5c0-1.381 1.119-2.5 2.5-2.5h15c1.381 0 2.5 1.119 2.5 2.5v13c0 1.381-1.119 2.5-2.5 2.5h-15c-1.381 0-2.5-1.119-2.5-2.5v-13zm2.5-.5c-.276 0-.5.224-.5.5v2.764l8 3.638 8-3.636V5.5c0-.276-.224-.5-.5-.5h-15zm15.5 5.463l-8 3.636-8-3.638V18.5c0 .276.224.5.5.5h15c.276 0 .5-.224.5-.5v-8.037zM10.9 14.2l-2.8 1.4.7-2.8 5.5-5.5 2.1 2.1-5.5 5.5zm7.3-6.6l-1.1-1.1c-.2-.2-.5-.2-.7 0l-.7.7 2.2 2.2.7-.7c.2-.2.2-.5 0-.7z"></path></g></svg>
         </div>
      </div>
    </div>

    <!-- Search & Filters -->
    <div class="p-4 border-b border-[#2F3336] flex flex-col gap-4">
        <!-- Search -->
        <div class="relative group w-full">
            <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 text-gray-500"><g><path d="M10.25 3.75c-3.59 0-6.5 2.91-6.5 6.5s2.91 6.5 6.5 6.5c1.795 0 3.419-.726 4.596-1.904 1.178-1.177 1.904-2.801 1.904-4.596 0-3.59-2.91-6.5-6.5-6.5zm-8.5 6.5c0-4.694 3.806-8.5 8.5-8.5s8.5 3.806 8.5 8.5c0 1.986-.73 3.815-1.945 5.232l4.944 4.942-1.414 1.415-4.942-4.944C14.065 18.02 12.236 18.75 10.25 18.75c-4.694 0-8.5-3.806-8.5-8.5z"></path></g></svg>
            </div>
            <input 
                id="messages-search-input"
                v-model="searchQuery" 
                @keydown.enter="handleSearch"
                type="text" 
                placeholder="Search Direct Messages" 
                class="w-full bg-[#202327] text-white rounded-full py-2 pl-10 pr-4 focus:outline-none focus:ring-1 focus:ring-[#1D9BF0] border border-transparent placeholder-gray-500 text-sm"
            >
        </div>

        <div class="flex flex-wrap items-center gap-4 text-sm text-[#71767B]">
             <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
                <input id="messages-filter-unread-checkbox" type="checkbox" v-model="filterUnread" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
                Unread
             </label>
             <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
                <input id="messages-filter-requests-checkbox" type="checkbox" v-model="filterRequests" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
                Requests
             </label>

             <!-- Sort -->
             <div class="relative">
                <div id="messages-sort-dropdown" @click="showSortDropdown = !showSortDropdown" class="flex items-center gap-1 cursor-pointer hover:text-white">
                    <span>{{ sortOption === 'unread_first' ? 'Unread First' : 'Latest' }}</span>
                    <svg viewBox="0 0 24 24" aria-hidden="true" class="h-4 w-4 fill-current"><g><path d="M3.543 8.96l1.414-1.42L12 14.59l7.043-7.05 1.414 1.42L12 17.41 3.543 8.96z"></path></g></svg>
                </div>
                <div v-if="showSortDropdown" class="absolute top-full left-0 mt-2 bg-black border border-[#2F3336] rounded-lg shadow-xl z-50 py-2 w-36">
                    <div id="messages-sort-unread-first" @click="handleSort('unread_first')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Unread First</div>
                    <div id="messages-sort-latest" @click="handleSort('latest')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Latest</div>
                </div>
             </div>
        </div>
    </div>

    <!-- Thread List -->
    <div id="messages-thread-list-container" class="flex flex-col divide-y divide-[#2F3336]">
       <div id="messages-thread-list">
          <div v-if="filteredThreads.length === 0" class="p-8 text-center text-[#71767B]">
              No messages found.
          </div>
          
          <div 
             v-for="thread in filteredThreads" 
             :key="thread.id" 
             :class="getThreadClass(thread)"
             class="p-4 hover:bg-white/[0.03] transition-colors cursor-pointer flex items-center gap-3"
             @click="handleOpenThread(thread)"
          >
             <div class="w-12 h-12 rounded-full overflow-hidden bg-gray-700 flex-shrink-0">
                 <img :src="getParticipant(thread.participant_id)?.avatar || '/images/photo1766328764.jpg'" alt="avatar" class="w-full h-full object-cover">
             </div>
             
             <div class="flex-1 min-w-0 flex flex-col">
                 <div class="flex items-center justify-between">
                     <div class="flex items-center gap-1 min-w-0">
                         <span class="font-bold text-white truncate">{{ getParticipant(thread.participant_id)?.name }}</span>
                         <span class="text-[#71767B] truncate">{{ getParticipant(thread.participant_id)?.handle }}</span>
                         <span class="text-[#71767B]">·</span>
                         <span class="text-[#71767B]">{{ thread.timestamp }}</span>
                     </div>
                 </div>
                 <div class="flex items-center justify-between text-[#71767B]">
                    <span class="truncate pr-4" :class="thread.unread ? 'font-bold text-white' : ''">
                        {{ thread.last_message }}
                    </span>
                    <div v-if="thread.unread" class="w-2 h-2 bg-[#1D9BF0] rounded-full flex-shrink-0"></div>
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
  name: 'MESSAGES_INBOX',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const searchQuery = ref('');
    const filterUnread = ref(false);
    const filterRequests = ref(false);
    const sortOption = ref(null);
    const showSortDropdown = ref(false);

    const filteredThreads = computed(() => {
        let result = [...dataStore.threads];

        if (signatureStore.matched_thread_id) {
            return result.filter(t => t.id === signatureStore.matched_thread_id);
        }
        if (searchQuery.value) {
           // handled in handleSearch action generally
        }

        if (filterUnread.value) {
            result = result.filter(t => t.unread);
        }
        if (filterRequests.value) {
            result = result.filter(t => t.is_request);
        }

        if (sortOption.value === 'unread_first') {
            result.sort((a, b) => (b.unread === a.unread) ? 0 : b.unread ? 1 : -1);
        } else if (sortOption.value === 'latest') {
            // mock sort
        }

        return result;
    });

    const getParticipant = (id) => dataStore.getUserById(id);

    const getThreadClass = (thread) => {
        const classes = [`data-id-${thread.id}`];
        if (signatureStore.matched_thread_id === thread.id) classes.push('thread-search-result');
        else if (filterUnread.value || filterRequests.value || sortOption.value) classes.push('thread-filtered');
        else classes.push('thread-visible');
        return classes.join(' ');
    };

    const handleSearch = () => {
        if (!searchQuery.value) return;
        const participantMatch = dataStore.users.find(u => u.name.toLowerCase().includes(searchQuery.value.toLowerCase()));
        
        // Find thread with matching participant OR matching message content (if we had full messages index, for now just participant or last_message)
        const match = dataStore.threads.find(t => {
            const participant = getParticipant(t.participant_id);
            return (participant && participant.name.toLowerCase().includes(searchQuery.value.toLowerCase())) || 
                   t.last_message.toLowerCase().includes(searchQuery.value.toLowerCase());
        });

        if (match) {
            signatureStore.matched_thread_id = match.id;
            signatureStore.messages_inbox_has_searched = true;
        }
    };

    const handleSort = (opt) => {
        sortOption.value = opt;
        signatureStore.messages_inbox_filters_applied = true;
        showSortDropdown.value = false;
    };

    const handleOpenThread = (thread) => {
        signatureStore.thread_id = thread.id;
        signatureStore.setCurrentPageId('MESSAGES_THREAD');
        signatureStore.messages_inbox_filters_applied = null;
        signatureStore.matched_thread_id = null;
        signatureStore.messages_inbox_has_searched = null;
        router.push({ name: 'MESSAGES_THREAD', params: { thread_id: thread.id } });
    };

    const handleNewMessage = () => {
        signatureStore.setCurrentPageId('MESSAGES_COMPOSE');
        router.push({ name: 'MESSAGES_COMPOSE' });
    };

    const handleBackHome = () => {
        signatureStore.setCurrentPageId('HOME');
        router.push({ name: 'HOME' });
    };
    
    watch([filterUnread, filterRequests, sortOption], () => {
       signatureStore.messages_inbox_filters_applied = true;
    });

    return {
        searchQuery,
        filterUnread,
        filterRequests,
        sortOption,
        showSortDropdown,
        filteredThreads,
        getParticipant,
        getThreadClass,
        handleSearch,
        handleSort,
        handleOpenThread,
        handleNewMessage,
        handleBackHome
    };
  }
}
</script>