<template>
  <div class="h-screen flex flex-col bg-gray-50">
    <!-- Header -->
    <header class="bg-white text-gray-800 p-4 shadow-sm border-b border-gray-200 flex justify-between items-center z-20">
      <div class="font-bold text-lg flex items-center">
        <button id="chat-back-home" @click="goHome" class="mr-4 hover:bg-gray-100 p-1 rounded">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        Chat
      </div>
      <div class="flex items-center gap-4">
        <!-- Search ACT_CHAT_LIST_SEARCH -->
        <div class="relative">
          <input 
            id="chat-search-input"
            type="text" 
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            placeholder="Search chats..."
            class="pl-10 pr-4 py-2 rounded bg-gray-100 text-gray-900 placeholder-gray-500 border-none focus:ring-2 focus:ring-[#6264A7] w-64"
          />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-500 absolute left-3 top-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>
    </header>

    <div class="flex-1 flex overflow-hidden">
      <!-- Sidebar Filters -->
      <aside class="w-64 bg-gray-50 border-r border-gray-200 p-4 flex flex-col gap-6 overflow-y-auto">
        <div>
          <h3 class="font-semibold text-gray-700 mb-2">Filters</h3>
          <!-- Checkbox Filter ACT_CHAT_LIST_FILTER_CHECKBOX -->
          <div class="flex items-center gap-2 mb-4">
            <input 
              id="filter-unread-checkbox"
              type="checkbox" 
              v-model="unreadOnly"
              class="w-4 h-4 text-[#6264A7] rounded focus:ring-[#6264A7]"
            />
            <label for="filter-unread-checkbox" class="text-sm text-gray-600">Unread only</label>
          </div>

          <!-- Slider Filter ACT_CHAT_LIST_FILTER_SLIDER -->
          <div class="mb-4">
            <label class="text-sm text-gray-600 block mb-1">Activity Level: > {{ minActivity }}%</label>
            <input 
              id="chat-activity-slider"
              type="range" 
              min="0" 
              max="100" 
              v-model.number="minActivity"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#6264A7]"
            />
          </div>
        </div>

        <div>
          <h3 class="font-semibold text-gray-700 mb-2">Sort By</h3>
          <!-- Sort Dropdown ACT_CHAT_LIST_FILTER_SORT -->
          <div id="chat-sort-dropdown" class="relative">
            <div 
              @click="toggleSort"
              class="w-full border rounded px-3 py-2 text-sm text-gray-700 bg-white cursor-pointer flex justify-between items-center"
            >
              {{ sortBy === 'recent' ? 'Most Recent' : (sortBy === 'name' ? 'Name (A-Z)' : 'Select...') }}
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>
            <div v-if="sortOpen" class="absolute top-full left-0 right-0 mt-1 bg-white border rounded shadow-lg z-10">
              <div id="chat-sort-recent" @click="setSort('recent')" class="px-3 py-2 text-sm hover:bg-gray-100 cursor-pointer">Most Recent</div>
              <div id="chat-sort-name-inc" @click="setSort('name')" class="px-3 py-2 text-sm hover:bg-gray-100 cursor-pointer">Name (A-Z)</div>
            </div>
          </div>
        </div>
      </aside>

      <!-- Main Content -->
      <main id="chat-list-container" class="flex-1 overflow-y-auto bg-white">
        
        <div id="chat-list" class="divide-y divide-gray-100">
          <div 
            v-for="chat in filteredChats" 
            :key="chat.id"
            :class="`data-id-${chat.id} p-4 hover:bg-gray-50 cursor-pointer flex items-center gap-4 group ${getChatClass(chat)}`"
            @click="openChat(chat)"
          >
            <div class="relative">
               <img 
                :src="chat.image" 
                class="w-12 h-12 rounded-full object-cover" 
                alt="Avatar"
                @error="$event.target.src = 'https://picsum.photos/100/100'"
               />
               <div v-if="chat.unread > 0" class="absolute -top-1 -right-1 bg-red-500 text-white text-xs w-5 h-5 flex items-center justify-center rounded-full border-2 border-white">
                 {{ chat.unread }}
               </div>
            </div>
            <div class="flex-1 min-w-0">
               <div class="flex justify-between items-baseline">
                 <h3 class="font-bold text-gray-900 group-hover:text-[#6264A7] truncate">{{ chat.name }}</h3>
                 <span class="text-xs text-gray-500">{{ chat.time }}</span>
               </div>
               <p class="text-sm text-gray-500 truncate" :class="{ 'font-semibold text-gray-800': chat.unread > 0 }">
                 {{ chat.lastMessage }}
               </p>
            </div>
          </div>
          
          <div v-if="filteredChats.length === 0" class="flex flex-col items-center justify-center p-12 text-gray-500 h-full">
             <p>No conversations found.</p>
          </div>
        </div>
      </main>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CHAT_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const unreadOnly = ref(false)
    const minActivity = ref(0)
    const sortBy = ref('')
    const sortOpen = ref(false)

    // Filter Logic
    const filteredChats = computed(() => {
      let result = dataStore.chats;

      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase();
        result = result.filter(c => c.name.toLowerCase().includes(q) || c.lastMessage.toLowerCase().includes(q));
      }

      if (unreadOnly.value) {
        result = result.filter(c => c.unread > 0);
      }

      if (minActivity.value > 0) {
        result = result.filter(c => c.activity > minActivity.value);
      }

      if (sortBy.value === 'name') {
        result = [...result].sort((a, b) => a.name.localeCompare(b.name));
      } else if (sortBy.value === 'recent') {
        // Mock simple sort
        result = [...result].sort((a, b) => b.activity - a.activity);
      }

      return result;
    })

    const handleSearch = () => {
      store.chat_list_has_searched = true;
      store.matched_chat_id = filteredChats.value.length > 0 ? filteredChats.value[0].id : null;
    }

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value
    }

    const setSort = (type) => {
      sortBy.value = type;
      sortOpen.value = false;
      store.chat_list_filters_applied = true;
    }

    const getChatClass = (chat) => {
      let classes = 'chat-row-visible ';
      if (store.chat_list_filters_applied) classes += 'chat-row-filtered ';
      if (store.chat_list_has_searched) classes += 'chat-row-matched ';
      return classes;
    }

    const openChat = async (chat) => {
      store.selected_chat_id = chat.id;
      // Clear flags
      store.chat_list_filters_applied = null;
      store.chat_list_has_searched = null;
      store.chat_list_viewport_anchor_id = null;
      
      store.currentPageId = 'CHAT_THREAD';
      await router.push({ name: 'CHAT_THREAD', params: { chatId: chat.id } });
    }

    const goHome = async () => {
      store.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    }

    return {
      searchQuery,
      unreadOnly,
      minActivity,
      sortBy,
      sortOpen,
      filteredChats,
      handleSearch,
      toggleSort,
      setSort,
      getChatClass,
      openChat,
      goHome,
      store
    }
  },
  watch: {
    unreadOnly() {
      this.store.chat_list_filters_applied = true;
    },
    minActivity() {
      this.store.chat_list_filters_applied = true;
    }
  }
}
</script>