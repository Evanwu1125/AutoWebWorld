<template>
  <div class="h-screen flex flex-col bg-white overflow-hidden">
    <!-- Header (Simplified) -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md z-20 justify-between shrink-0">
        <div class="font-semibold cursor-pointer" @click="goHome" id="back-home">Outlook Mail</div>
        <div class="flex gap-4">
             <div class="cursor-pointer hover:text-gray-200" id="folder-trash" @click="goToTrash">Trash</div>
        </div>
    </header>

    <div class="flex flex-1 overflow-hidden">
        <!-- Sidebar -->
        <div class="w-64 bg-gray-50 border-r border-gray-200 flex flex-col shrink-0">
             <div class="p-4">
                 <button id="button-new-message" class="w-full bg-[#0078D4] text-white py-2 px-4 rounded shadow-sm hover:bg-[#005A9E] font-medium flex items-center justify-center gap-2" @click="openCompose">
                    <span>+</span> New mail
                 </button>
             </div>
             <nav class="flex-1 overflow-y-auto">
                 <div class="px-2 py-1 text-sm font-semibold text-gray-500 uppercase tracking-wider mt-4 mb-2">Folders</div>
                 <div class="bg-blue-100 text-[#0078D4] font-medium px-4 py-2 cursor-pointer border-l-4 border-[#0078D4]">Inbox</div>
                 <div class="px-4 py-2 hover:bg-gray-100 cursor-pointer text-gray-700">Sent Items</div>
                 <div class="px-4 py-2 hover:bg-gray-100 cursor-pointer text-gray-700">Drafts</div>
                 <div class="px-4 py-2 hover:bg-gray-100 cursor-pointer text-gray-700" @click="goToTrash">Trash</div>
             </nav>
        </div>

        <!-- Mail List -->
        <div class="w-96 border-r border-gray-200 flex flex-col bg-white shrink-0">
            <!-- Toolbar -->
            <div class="p-3 border-b border-gray-200 flex flex-col gap-3">
                <!-- Search -->
                <div class="relative">
                     <input type="text" id="inbox-search-input" v-model="searchQuery" @keypress.enter="handleSearch" placeholder="Search Inbox" class="w-full border border-gray-300 rounded px-3 py-1.5 text-sm focus:outline-none focus:border-[#0078D4]" />
                     <button @click="handleSearch" class="absolute right-2 top-1.5 text-gray-400 hover:text-[#0078D4]">🔍</button>
                </div>
                
                <!-- Filters -->
                <div class="flex items-center justify-between text-sm">
                    <div class="flex items-center gap-2">
                        <label class="flex items-center gap-1 cursor-pointer select-none">
                            <input type="checkbox" id="filter-unread-checkbox" @change="handleFilterCheckbox" class="rounded text-[#0078D4] focus:ring-[#0078D4]" />
                            <span>Unread</span>
                        </label>
                    </div>
                    
                    <!-- Sort Dropdown -->
                    <div class="relative">
                        <div id="sort-dropdown" class="flex items-center gap-1 cursor-pointer hover:text-[#0078D4]" @click="toggleSortMenu">
                             <span>Sort</span>
                             <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                        </div>
                        <div v-if="showSortMenu" class="absolute right-0 top-6 bg-white shadow-lg border border-gray-200 rounded z-50 w-32 py-1">
                             <div id="sort-option-newest-desc" class="px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="handleSort('newest')">Newest</div>
                             <div id="sort-option-oldest" class="px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="handleSort('oldest')">Oldest</div>
                             <div id="sort-option-sender" class="px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="handleSort('by_sender')">By Sender</div>
                        </div>
                    </div>
                </div>

                <!-- Slider Filter (Size) -->
                <div class="flex items-center gap-2 text-xs text-gray-600 px-1">
                     <span>Size:</span>
                     <input type="range" id="size-slider" min="0" max="1500" step="1" v-model="sizeFilter" @input="handleSliderChange" class="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#0078D4]" />
                </div>
            </div>

            <!-- Email Items -->
            <div id="mail-list" class="flex-1 overflow-y-auto" @scroll="handleScroll">
                 <div v-for="email in displayEmails" :key="email.id" 
                      :class="['p-4 border-b border-gray-100 cursor-pointer hover:bg-[#F3F2F1]', 
                               {'bg-[#E1DFDD]': email.id === selectedEmailId},
                               'row-visible',
                               isFiltered ? 'row-filtered' : '',
                               isSearched ? 'row-search-result' : '',
                               `data-id-${email.id}`]"
                      @click="openEmail(email)">
                      
                      <div class="flex justify-between items-start mb-1">
                          <div class="font-semibold text-gray-900 truncate pr-2">{{ email.sender }}</div>
                          <div class="text-xs text-gray-500 whitespace-nowrap">{{ email.time }}</div>
                      </div>
                      <div class="text-[#0078D4] text-sm font-medium mb-1 truncate">{{ email.subject }}</div>
                      <div class="text-gray-500 text-xs truncate">{{ email.preview }}</div>
                 </div>
                 
                 <!-- Empty State -->
                 <div v-if="displayEmails.length === 0" class="p-8 text-center text-gray-500">
                     <div class="text-4xl mb-2">📭</div>
                     <div>Nothing to show here</div>
                 </div>
            </div>
        </div>

        <!-- Reading Pane Placeholder (Responsive hidden on mobile usually, but here fixed layout) -->
        <div class="flex-1 bg-gray-50 flex items-center justify-center text-gray-400">
            <div class="text-center">
                <svg class="w-24 h-24 mx-auto mb-4 opacity-20" fill="currentColor" viewBox="0 0 20 20"><path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" /><path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" /></svg>
                <div>Select an item to read</div>
            </div>
        </div>
    </div>

    <!-- Permission Modal (Location) -->
    <div v-if="showLocationModal" class="fixed inset-0 bg-black/50 backdrop-blur-sm z-[9999] flex items-center justify-center p-4">
        <div class="bg-white rounded-lg shadow-xl p-6 max-w-sm w-full">
            <div class="text-center mb-4">
                <div class="w-12 h-12 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-3">📍</div>
                <h3 class="text-lg font-bold text-gray-900">Location Access Required</h3>
                <p class="text-gray-600 text-sm mt-2">
                    Outlook needs access to your location to provide better service and local weather updates.
                </p>
            </div>
            <div class="flex gap-3">
                <button class="flex-1 py-2 border border-gray-300 rounded text-gray-700 hover:bg-gray-50 font-medium transition-colors">Deny</button>
                <button id="permission-location-allow" class="flex-1 py-2 bg-[#0078D4] text-white rounded hover:bg-[#005A9E] font-medium transition-colors shadow-sm" @click="grantLocation">Allow Access</button>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'MAIL_INBOX',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();
    
    const searchQuery = ref('');
    const showSortMenu = ref(false);
    const sizeFilter = ref(0);
    const isUnreadFilter = ref(false);
    const sortOption = ref(null);
    const showLocationModal = ref(false);
    
    // Derived state flags for FSM
    const isFiltered = computed(() => isUnreadFilter.value || sizeFilter.value > 0 || sortOption.value !== null);
    const isSearched = computed(() => searchQuery.value.length > 0);
    const selectedEmailId = computed(() => signatureStore.selected_email_id);

    onMounted(() => {
        // Check for location permission requirement
        if (signatureStore.location_permission_granted === null) {
            showLocationModal.value = true;
        }
    });

    const displayEmails = computed(() => {
        let emails = dataStore.emails || [];
        
        // Search
        if (searchQuery.value) {
            const query = searchQuery.value.toLowerCase();
            emails = emails.filter(e => 
                e.subject.toLowerCase().includes(query) || 
                e.sender.toLowerCase().includes(query) ||
                e.preview.toLowerCase().includes(query)
            );
        }

        // Filter: Unread
        if (isUnreadFilter.value) {
            emails = emails.filter(e => !e.read);
        }

        // Filter: Size (Mock logic: size > filter value)
        if (sizeFilter.value > 0) {
            emails = emails.filter(e => (e.size || 0) >= sizeFilter.value);
        }

        // Sort
        if (sortOption.value === 'newest') {
            emails = [...emails].sort((a, b) => new Date(b.date) - new Date(a.date));
        } else if (sortOption.value === 'oldest') {
            emails = [...emails].sort((a, b) => new Date(a.date) - new Date(b.date));
        } else if (sortOption.value === 'by_sender') {
            emails = [...emails].sort((a, b) => a.sender.localeCompare(b.sender));
        }

        return emails;
    });

    const grantLocation = () => {
        signatureStore.handleAction('ACT_INBOX_GRANT_LOCATION');
        showLocationModal.value = false;
    };

    const handleFilterCheckbox = (e) => {
        isUnreadFilter.value = e.target.checked;
        signatureStore.handleAction('ACT_INBOX_FILTER_CHECKBOX', { widget: 'checkboxes' });
    };

    const handleSliderChange = () => {
        signatureStore.handleAction('ACT_INBOX_FILTER_SLIDER', { widget: 'sliders' });
    };

    const toggleSortMenu = () => {
        showSortMenu.value = !showSortMenu.value;
    };

    const handleSort = (option) => {
        sortOption.value = option;
        showSortMenu.value = false;
        signatureStore.handleAction('ACT_INBOX_FILTER_SORT', { widget: 'sort' });
    };

    const handleSearch = () => {
        signatureStore.handleAction('ACT_INBOX_SEARCH_EMAIL', { search_query: searchQuery.value, item_id: 'search-result-1' }); // Mock ID
    };

    const openEmail = async (email) => {
        // Determine which action to call based on state
        if (isSearched.value) {
            await signatureStore.handleAction('ACT_INBOX_OPEN_MATCHED_EMAIL', { item_id: email.id });
        } else if (isFiltered.value) {
            await signatureStore.handleAction('ACT_INBOX_OPEN_FILTERED_EMAIL', { item_id: email.id });
        } else {
            // For ACT_INBOX_OPEN_ANY_EMAIL, we first need to set anchor ID via scroll/hover logic in FSM
            // But for simplicity in UI action mapping, we'll set the anchor then open
            // In a strict FSM runner, scroll happens first. Here we assume user clicked visible row.
            // We'll manually set the precondition state for "OPEN_ANY" to succeed
            signatureStore.inbox_viewport_anchor_id = email.id; 
            await signatureStore.handleAction('ACT_INBOX_OPEN_ANY_EMAIL', { item_id: email.id });
        }
        router.push({ name: 'MAIL_MESSAGE_READ', params: { id: email.id } });
    };

    const handleScroll = () => {
        // Debounce logic could go here
        // Simulating scroll into view action
        if (displayEmails.value.length > 0) {
             signatureStore.handleAction('ACT_INBOX_SCROLL_EMAIL_INTO_VIEW', { item_id: displayEmails.value[0].id });
        }
    };

    const openCompose = async () => {
        await signatureStore.handleAction('ACT_INBOX_OPEN_COMPOSE');
        router.push({ name: 'MAIL_COMPOSE' });
    };

    const goToTrash = async () => {
        await signatureStore.handleAction('ACT_INBOX_GO_TRASH');
        router.push({ name: 'MAIL_TRASH' });
    };

    const goHome = async () => {
        await signatureStore.handleAction('ACT_INBOX_BACK_HOME');
        router.push({ name: 'HOME' });
    };

    return {
        searchQuery,
        showSortMenu,
        sizeFilter,
        showLocationModal,
        displayEmails,
        selectedEmailId,
        isFiltered,
        isSearched,
        grantLocation,
        handleFilterCheckbox,
        handleSliderChange,
        toggleSortMenu,
        handleSort,
        handleSearch,
        openEmail,
        handleScroll,
        openCompose,
        goToTrash,
        goHome
    };
  }
}
</script>