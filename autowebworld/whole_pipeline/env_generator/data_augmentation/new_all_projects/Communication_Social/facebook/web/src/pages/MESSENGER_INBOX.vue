<template>
  <div class="min-h-screen bg-gray-100 pb-10">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20 h-16 flex items-center px-4 justify-between">
      <div class="flex items-center gap-4">
        <button 
          id="messenger-back-home"
          @click="goBack"
          class="p-2 hover:bg-gray-100 rounded-full transition-colors"
        >
          <svg class="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        <h1 class="text-xl font-bold text-gray-900">Chats</h1>
      </div>
      <button 
        id="new-message-button"
        @click="newMessage"
        class="p-2 bg-gray-100 hover:bg-gray-200 rounded-full transition-colors text-blue-600"
      >
        <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
        </svg>
      </button>
    </header>

    <div class="max-w-2xl mx-auto px-4 py-6">
      <!-- Filters -->
      <div class="flex items-center justify-between mb-4">
         <div class="flex items-center gap-2">
            <label class="flex items-center gap-2 cursor-pointer bg-white px-3 py-1.5 rounded-full shadow-sm border border-gray-200 hover:bg-gray-50 transition-colors">
              <div 
                id="filter-unread-checkbox"
                class="w-4 h-4 border border-gray-400 rounded-full flex items-center justify-center transition-colors"
                :class="{ 'bg-blue-600 border-blue-600': filters.unreadOnly }"
                @click.prevent="toggleUnread"
              ></div>
              <span class="text-sm font-medium text-gray-700">Unread</span>
            </label>
         </div>

         <!-- Sort -->
         <div class="relative">
            <button 
              id="messenger-sort-dropdown"
              @click="toggleSort"
              class="flex items-center gap-1 text-gray-600 text-sm font-medium hover:text-gray-900"
            >
              Sort: {{ sortOption === 'recent' ? 'Recent' : 'Unread First' }}
              <svg class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            <div v-if="sortOpen" class="absolute right-0 mt-2 w-40 bg-white rounded-md shadow-lg py-1 z-10 ring-1 ring-black ring-opacity-5">
              <div 
                id="messenger-sort-option-recent"
                @click="selectSort('recent')"
                class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
              >
                Recent
              </div>
              <div 
                id="messenger-sort-option-unread"
                @click="selectSort('unread')"
                class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
              >
                Unread First
              </div>
            </div>
          </div>
      </div>

      <!-- Thread List -->
      <div id="messenger-thread-list" class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden divide-y divide-gray-100">
        <div 
          v-for="thread in filteredThreads" 
          :key="thread.id" 
          class="p-4 flex items-center gap-4 hover:bg-gray-50 cursor-pointer transition-colors relative"
          :class="{ 'thread-visible': true, 'thread-filtered': isFiltered, 'bg-blue-50': thread.unread }"
          :data-id-value="thread.id"
          @click="openThread(thread)"
        >
          <div class="relative flex-shrink-0">
             <img :src="thread.avatar" class="h-14 w-14 rounded-full object-cover" :alt="thread.name" />
             <span v-if="thread.unread" class="absolute bottom-0 right-0 block h-3.5 w-3.5 rounded-full ring-2 ring-white bg-blue-600"></span>
          </div>
          
          <div class="flex-1 min-w-0">
            <div class="flex justify-between items-baseline mb-1">
               <h3 class="text-base font-semibold text-gray-900 truncate">{{ thread.name }}</h3>
               <span class="text-xs text-gray-500 whitespace-nowrap ml-2">{{ thread.time }}</span>
            </div>
            <p 
                class="text-sm truncate"
                :class="{ 'font-bold text-gray-900': thread.unread, 'text-gray-500': !thread.unread }"
            >
               <span v-if="thread.id === 'thread_1'" :class="`data-id-${thread.id}`">{{ thread.last_message }}</span>
               <span v-else :class="`data-id-${thread.id}`">{{ thread.last_message }}</span>
            </p>
          </div>
        </div>
        
        <div v-if="filteredThreads.length === 0" class="text-center py-10">
            <p class="text-gray-500">No conversations found.</p>
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
import { orderBy } from 'lodash-es';

export default {
  name: 'MESSENGER_INBOX',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const sortOpen = ref(false);
    const sortOption = ref('recent');
    const filters = ref({
      unreadOnly: false
    });

    const isFiltered = computed(() => {
      return filters.value.unreadOnly;
    });

    const filteredThreads = computed(() => {
      let result = [...dataStore.threads];

      if (filters.value.unreadOnly) {
        result = result.filter(t => t.unread);
      }

      if (sortOption.value === 'unread') {
        result = orderBy(result, ['unread', 'id'], ['desc', 'asc']); // Unread first
      } else {
        // Recent -> By ID (mock chronological)
        result = orderBy(result, ['id'], ['asc']);
      }

      return result;
    });

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value;
    };

    const selectSort = (option) => {
      sortOption.value = option;
      sortOpen.value = false;
      signatureStore.messenger_inbox_filters_applied = true; // FSM Effect
    };

    const toggleUnread = () => {
      filters.value.unreadOnly = !filters.value.unreadOnly;
      signatureStore.messenger_inbox_filters_applied = true; // FSM Effect
    };

    const openThread = async (thread) => {
      signatureStore.selected_thread_id = thread.id;
      // Clear anchor
      signatureStore.messenger_inbox_viewport_anchor_id = null;
      // Clear filters
      if (isFiltered.value) {
        signatureStore.messenger_inbox_filters_applied = null;
      }
      
      await router.push({ name: 'MESSAGE_THREAD', params: { id: thread.id } });
    };

    const newMessage = async () => {
      signatureStore.currentPageId = 'MESSAGE_COMPOSE';
      await router.push({ name: 'MESSAGE_COMPOSE' });
    };

    const goBack = async () => {
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    return {
      sortOpen,
      sortOption,
      filters,
      isFiltered,
      filteredThreads,
      toggleSort,
      selectSort,
      toggleUnread,
      openThread,
      newMessage,
      goBack
    };
  }
}
</script>