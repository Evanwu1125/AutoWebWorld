<template>
  <div class="flex flex-col min-h-screen bg-black text-white pb-20 sm:pb-0">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md px-4 py-3 flex items-center gap-4 border-b border-[#2F3336]">
      <div id="following-back-profile" @click="handleBack" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
         <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M7.414 13l5.043 5.04-1.414 1.42L3.586 12l7.457-7.46 1.414 1.42L7.414 11H21v2H7.414z"></path></g></svg>
      </div>
      <div>
         <h2 class="text-xl font-bold">Following</h2>
      </div>
    </div>

    <!-- Filters -->
    <div class="p-4 border-b border-[#2F3336] flex flex-wrap gap-4 text-sm text-[#71767B]">
        <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
            <input id="following-filter-verified-checkbox" type="checkbox" v-model="filterVerified" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
            Verified
        </label>
        <label class="flex items-center gap-2 cursor-pointer hover:text-white transition-colors">
            <input id="following-filter-muted-checkbox" type="checkbox" v-model="filterMuted" class="form-checkbox bg-transparent border-[#536471] text-[#1D9BF0] rounded focus:ring-0 focus:ring-offset-0">
            Muted
        </label>

        <!-- Sort -->
        <div class="relative">
            <div id="following-sort-dropdown" @click="showSortDropdown = !showSortDropdown" class="flex items-center gap-1 cursor-pointer hover:text-white">
                <span>{{ sortOption === 'latest' ? 'Latest' : 'Alphabetical' }}</span>
                <svg viewBox="0 0 24 24" aria-hidden="true" class="h-4 w-4 fill-current"><g><path d="M3.543 8.96l1.414-1.42L12 14.59l7.043-7.05 1.414 1.42L12 17.41 3.543 8.96z"></path></g></svg>
            </div>
            <div v-if="showSortDropdown" class="absolute top-full left-0 mt-2 bg-black border border-[#2F3336] rounded-lg shadow-xl z-50 py-2 w-36">
                <div id="following-sort-desc" @click="handleSort('latest')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Latest</div>
                <div id="following-sort-alphabetical-inc" @click="handleSort('alphabetical')" class="px-4 py-2 hover:bg-white/10 cursor-pointer text-white">Alphabetical</div>
            </div>
        </div>
    </div>

    <!-- List -->
    <div id="following-list-container" class="flex flex-col divide-y divide-[#2F3336]">
       <div id="following-list">
          <div v-if="filteredUsers.length === 0" class="p-8 text-center text-[#71767B]">
              No following users found.
          </div>
          
          <div 
             v-for="user in filteredUsers" 
             :key="user.id" 
             :class="getUserClass(user)"
             class="p-4 hover:bg-white/[0.03] transition-colors cursor-pointer flex items-center justify-between"
             @click="handleOpenUser(user)"
          >
             <div class="flex items-center gap-3">
                 <div class="w-12 h-12 rounded-full overflow-hidden bg-gray-700">
                     <img :src="user.avatar || '/images/photo1766328617.jpg'" alt="avatar" class="w-full h-full object-cover">
                 </div>
                 <div class="flex flex-col">
                     <div class="font-bold text-white flex items-center gap-1">
                         {{ user.name }}
                         <svg v-if="user.verified" viewBox="0 0 24 24" aria-hidden="true" class="h-4 w-4 text-[#1D9BF0] fill-current"><g><path d="M22.5 12.5c0-1.58-.875-2.95-2.148-3.6.154-.435.238-.905.238-1.4 0-2.21-1.71-3.998-3.818-3.998-.47 0-.92.084-1.336.25C14.818 2.415 13.51 1.5 12 1.5s-2.816.917-3.437 2.25c-.415-.165-.866-.25-1.336-.25-2.11 0-3.818 1.79-3.818 4 0 .495.083.965.238 1.4-1.272.65-2.147 2.018-2.147 3.6 0 1.495.782 2.798 1.942 3.486-.02.17-.032.34-.032.514 0 2.21 1.708 4 3.818 4 .47 0 .92-.086 1.335-.25.62 1.334 1.926 2.25 3.437 2.25 1.512 0 2.818-.916 3.437-2.25.415.163.865.248 1.336.248 2.11 0 3.818-1.79 3.818-4 0-.174-.012-.344-.033-.513 1.158-.687 1.943-1.99 1.943-3.484zm-6.616-3.334l-4.334 6.5c-.145.217-.382.334-.625.334-.143 0-.288-.04-.416-.126l-.115-.094-2.415-2.415c-.293-.293-.293-.768 0-1.06s.768-.294 1.06 0l1.77 1.767 3.825-5.74c.23-.345.696-.436 1.04-.207.346.23.44.696.21 1.04z"></path></g></svg>
                     </div>
                     <div class="text-[#71767B]">{{ user.handle }}</div>
                     <div class="text-white text-sm mt-1 line-clamp-1">{{ user.bio }}</div>
                 </div>
             </div>
             <div>
                 <button class="border border-[#536471] text-white font-bold rounded-full px-4 py-1.5 hover:bg-white/10 transition-colors">
                     Following
                 </button>
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
  name: 'PROFILE_FOLLOWING_LIST',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const filterVerified = ref(false);
    const filterMuted = ref(false);
    const sortOption = ref(null);
    const showSortDropdown = ref(false);

    // Mock following list: All users except me
    const filteredUsers = computed(() => {
        let result = dataStore.users.filter(u => u.id !== 'user_me');

        if (filterVerified.value) {
            result = result.filter(u => u.verified);
        }
        if (filterMuted.value) {
            // Mock logic for muted, e.g. based on id length is even
            result = result.filter(u => u.id.length % 2 === 0);
        }

        if (sortOption.value === 'alphabetical') {
            result.sort((a, b) => a.name.localeCompare(b.name));
        }

        return result;
    });

    const getUserClass = (user) => {
        const classes = [`data-id-${user.id}`];
        if (filterVerified.value || filterMuted.value || sortOption.value) classes.push('user-filtered');
        else classes.push('user-visible');
        return classes.join(' ');
    };

    const handleSort = (opt) => {
        sortOption.value = opt;
        signatureStore.profile_following_filters_applied = true;
        showSortDropdown.value = false;
    };

    const handleOpenUser = (user) => {
        signatureStore.user_id = user.id;
        signatureStore.setCurrentPageId('USER_PROFILE_OVERVIEW');
        signatureStore.profile_following_filters_applied = null;
        router.push({ name: 'USER_PROFILE_OVERVIEW', params: { user_id: user.id } });
    };

    const handleBack = () => {
        signatureStore.setCurrentPageId('PROFILE_OVERVIEW');
        router.push({ name: 'PROFILE_OVERVIEW' });
    };
    
    watch([filterVerified, filterMuted, sortOption], () => {
       signatureStore.profile_following_filters_applied = true;
    });

    return {
        filterVerified,
        filterMuted,
        sortOption,
        showSortDropdown,
        filteredUsers,
        getUserClass,
        handleSort,
        handleOpenUser,
        handleBack
    };
  }
}
</script>