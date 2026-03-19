<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans">
    <!-- Header -->
    <header class="bg-[#161b22] border-b border-gray-700 py-4 px-6 flex items-center gap-4">
      <div id="profile-back-home" class="cursor-pointer flex items-center gap-1 text-blue-400 hover:underline" @click="goHome">
          <svg height="16" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-left fill-current"><path d="M7.78 12.53a.75.75 0 0 1-1.06 0L2.47 8.28a.75.75 0 0 1 0-1.06l4.25-4.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L4.81 7h7.44a.75.75 0 0 1 0 1.5H4.81l2.97 2.97a.75.75 0 0 1 0 1.06Z"></path></svg>
          Back to Home
      </div>
    </header>

    <main class="max-w-6xl mx-auto p-6 grid grid-cols-1 md:grid-cols-4 gap-8">
        <!-- Left Sidebar: Profile Info -->
        <div class="space-y-6">
            <div class="relative group">
                <img :src="user.avatar" class="w-64 h-64 rounded-full border border-gray-700 mx-auto md:mx-0" />
                <div class="absolute bottom-2 left-0 bg-gray-800/80 p-1 rounded-md text-xs border border-gray-600 ml-6 md:ml-0">
                    Use ImageGetter to replace avatars
                </div>
            </div>
            
            <div>
                <h1 class="text-2xl font-bold">{{ user.name }}</h1>
                <h2 class="text-xl text-gray-400 font-light">{{ user.username }}</h2>
            </div>

            <button id="profile-settings-link" @click="goToSettings" class="w-full px-3 py-1.5 bg-[#21262d] border border-gray-600 rounded-md text-sm font-medium text-gray-300 hover:bg-gray-700">
                Edit profile
            </button>

            <div class="space-y-2 text-sm text-gray-300">
                <div class="flex items-center gap-2" v-if="user.bio">
                     <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                     {{ user.bio }}
                </div>
                <div class="flex items-center gap-2" v-if="user.location">
                     <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                     {{ user.location }}
                </div>
                <div class="flex items-center gap-2" v-if="user.website">
                     <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"></path></svg>
                     <a :href="user.website" class="text-blue-400 hover:underline">{{ user.website }}</a>
                </div>
            </div>

            <div class="flex gap-4 text-sm text-gray-400" id="profile-followers-link" @click="goToFollowers">
                 <span class="hover:text-blue-400 cursor-pointer flex items-center gap-1">
                     <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
                     <strong class="text-white">1.5k</strong> followers
                 </span>
                 <span class="hover:text-blue-400 cursor-pointer">
                     <strong class="text-white">42</strong> following
                 </span>
            </div>
        </div>

        <!-- Main Content -->
        <div class="col-span-3 space-y-6">
             <!-- Tabs -->
             <div class="border-b border-gray-700 pb-0 flex space-x-6 text-sm">
                 <div class="pb-3 border-b-2 border-[#fd8c73] font-semibold flex items-center gap-2 cursor-pointer">
                     <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-book"><path d="M0 5.75C0 4.784.784 4 1.75 4h12.5c.966 0 1.75.784 1.75 1.75v8.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h12.5a.25.25 0 0 0 .25-.25v-8.5a.25.25 0 0 0-.25-.25ZM3.5 6.25a.75.75 0 0 1 .75.75v6a.75.75 0 0 1-1.5 0v-6a.75.75 0 0 1 .75-.75Zm4.25.75a.75.75 0 0 0-1.5 0v6a.75.75 0 0 0 1.5 0ZM12 7a.75.75 0 0 0-1.5 0v6a.75.75 0 0 0 1.5 0Z"></path></svg>
                     Overview
                 </div>
                 <div id="profile-repos-link" @click="goToRepos" class="pb-3 border-b-2 border-transparent hover:border-gray-300 text-gray-300 flex items-center gap-2 cursor-pointer">
                     <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo"><path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 1 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 0 1 1-1ZM5 12.25a.25.25 0 0 1 .25-.25h3.5a.25.25 0 0 1 .25.25v3.25a.25.25 0 0 1-.4.2l-1.45-1.087a.25.25 0 0 0-.3 0L5.4 15.7a.25.25 0 0 1-.4-.2Z"></path></svg>
                     Repositories
                 </div>
             </div>

             <!-- Pinned Repos (Mock) -->
             <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                 <div class="p-4 border border-gray-700 rounded-md bg-[#0d1117] hover:bg-[#161b22] cursor-pointer flex flex-col justify-between">
                     <div>
                         <div class="flex items-center gap-2 mb-2">
                             <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo text-gray-400"><path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 1 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 0 1 1-1ZM5 12.25a.25.25 0 0 1 .25-.25h3.5a.25.25 0 0 1 .25.25v3.25a.25.25 0 0 1-.4.2l-1.45-1.087a.25.25 0 0 0-.3 0L5.4 15.7a.25.25 0 0 1-.4-.2Z"></path></svg>
                             <span class="font-bold text-blue-400">sample-repo</span>
                         </div>
                         <p class="text-xs text-gray-400">A sample repository for testing</p>
                     </div>
                     <div class="mt-4 text-xs text-gray-500 flex items-center gap-4">
                         <div class="flex items-center gap-1">
                            <span class="w-3 h-3 rounded-full bg-yellow-400"></span> JavaScript
                         </div>
                         <div>⭐ 1.2k</div>
                     </div>
                 </div>
                  <div class="p-4 border border-gray-700 rounded-md bg-[#0d1117] hover:bg-[#161b22] cursor-pointer flex flex-col justify-between">
                     <div>
                         <div class="flex items-center gap-2 mb-2">
                             <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-repo text-gray-400"><path d="M2 2.5A2.5 2.5 0 0 1 4.5 0h8.75a.75.75 0 0 1 .75.75v12.5a.75.75 0 0 1-.75.75h-2.5a.75.75 0 1 1 0-1.5h1.75v-2h-8a1 1 0 0 0-.714 1.7.75.75 0 1 1-1.072 1.05A2.495 2.495 0 0 1 2 11.5Zm10.5-1V9h-8c-.356 0-.694.074-1 .208V2.5a1 1 0 0 1 1-1ZM5 12.25a.25.25 0 0 1 .25-.25h3.5a.25.25 0 0 1 .25.25v3.25a.25.25 0 0 1-.4.2l-1.45-1.087a.25.25 0 0 0-.3 0L5.4 15.7a.25.25 0 0 1-.4-.2Z"></path></svg>
                             <span class="font-bold text-blue-400">vue-template</span>
                         </div>
                         <p class="text-xs text-gray-400">Vue 3 template for FSM apps</p>
                     </div>
                     <div class="mt-4 text-xs text-gray-500 flex items-center gap-4">
                         <div class="flex items-center gap-1">
                            <span class="w-3 h-3 rounded-full bg-green-400"></span> Vue
                         </div>
                         <div>⭐ 42</div>
                     </div>
                 </div>
             </div>

             <!-- Calendar (Mock) -->
             <div class="border border-gray-700 rounded-md p-4 bg-[#0d1117]">
                 <h3 class="text-sm font-semibold mb-4">1,234 contributions in the last year</h3>
                 <div class="grid grid-cols-53 gap-1 h-32">
                     <!-- Simple mock grid visual -->
                     <div v-for="i in 364" :key="i" class="w-full h-full rounded-[2px]" 
                        :class="['bg-[#161b22]', Math.random() > 0.8 ? 'bg-[#0e4429]' : '', Math.random() > 0.95 ? 'bg-[#39d353]' : '']"
                     ></div>
                 </div>
             </div>
        </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'PROFILE_OVERVIEW',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const dataStore = useDataStore();

    const user = computed(() => {
        // Mock current user to user_1 for profile demo
        return dataStore.users.find(u => u.id === 'user_1') || {};
    });

    const goHome = async () => {
        store.setCurrentPageId('HOME');
        await router.push({ name: 'HOME' });
    };

    const goToRepos = async () => {
        store.setCurrentPageId('REPOSITORIES_LIST');
        await router.push({ name: 'REPOSITORIES_LIST' });
    };

    const goToSettings = async () => {
        store.setCurrentPageId('PROFILE_SETTINGS');
        await router.push({ name: 'PROFILE_SETTINGS' });
    };

    const goToFollowers = async () => {
        store.setCurrentPageId('PROFILE_FOLLOWERS');
        await router.push({ name: 'PROFILE_FOLLOWERS' });
    };

    return {
        user,
        goHome,
        goToRepos,
        goToSettings,
        goToFollowers
    };
  }
}
</script>