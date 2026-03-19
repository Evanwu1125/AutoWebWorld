<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans">
    <!-- Header -->
    <header class="bg-[#161b22] border-b border-gray-700 py-4 px-6 flex items-center gap-4">
      <div id="followers-back-profile" class="cursor-pointer flex items-center gap-1 text-blue-400 hover:underline" @click="goBackProfile">
          <svg height="16" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-arrow-left fill-current"><path d="M7.78 12.53a.75.75 0 0 1-1.06 0L2.47 8.28a.75.75 0 0 1 0-1.06l4.25-4.25a.751.751 0 0 1 1.042.018.751.751 0 0 1 .018 1.042L4.81 7h7.44a.75.75 0 0 1 0 1.5H4.81l2.97 2.97a.75.75 0 0 1 0 1.06Z"></path></svg>
          Back to Profile
      </div>
      <div id="followers-back-home" class="cursor-pointer flex items-center gap-1 text-gray-400 hover:text-blue-400 hover:underline ml-auto" @click="goHome">
          Home
      </div>
    </header>

    <main class="max-w-4xl mx-auto p-6">
        <h1 class="text-2xl font-normal mb-6 border-b border-gray-700 pb-4">Followers</h1>
        
        <div id="followers-list" class="space-y-4">
            <div v-for="follower in followers" :key="follower.id" class="flex items-center justify-between p-4 border border-gray-700 rounded-md bg-[#161b22] hover:bg-[#1c2128]">
                <div class="flex items-center gap-4">
                    <img :src="follower.avatar" class="w-12 h-12 rounded-full border border-gray-700" />
                    <div>
                        <h3 class="font-bold text-lg hover:text-blue-400 cursor-pointer follower-row-visible" @click="openUser(follower)">
                            {{ follower.name }}
                        </h3>
                        <p class="text-gray-400 text-sm">{{ follower.username }}</p>
                        <div class="text-xs text-gray-500 mt-1 flex items-center gap-2">
                            <span v-if="follower.location">
                                <svg class="w-3 h-3 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                                {{ follower.location }}
                            </span>
                        </div>
                    </div>
                </div>
                <button class="px-3 py-1 text-xs font-medium text-gray-300 bg-[#21262d] border border-gray-600 rounded-md hover:bg-gray-700">
                    Unfollow
                </button>
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
import { FSMRuntime } from '../fsm/FSMRuntime';
import fsmData from '../../fsm.json';

export default {
  name: 'PROFILE_FOLLOWERS',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const dataStore = useDataStore();
    const fsmRuntime = new FSMRuntime(fsmData, { store, router });

    // Mock: Current user is user_1
    const currentUserId = 'user_1';

    const followers = computed(() => {
        // Find relationships where follower follows user_1
        const rels = dataStore.followers.filter(f => f.user_id === currentUserId);
        // Map to user objects
        return rels.map(r => dataStore.users.find(u => u.id === r.follower_id)).filter(Boolean);
    });

    const openUser = async (user) => {
        const action = fsmData.pages.find(p => p.id === 'PROFILE_FOLLOWERS').actions.find(a => a.id === 'ACT_FOLLOWERS_OPEN_ANY');
        if (action) {
             const params = { item_id: user.id };
             const nextSig = fsmRuntime.applyEffects(action, store.signature, params);
             store.signature.followers_selected_user_id = nextSig.followers_selected_user_id;
             store.signature.followers_viewport_anchor_id = null; // Clear anchor

             // Logic: usually opens that user's profile. Reusing PROFILE_OVERVIEW for simplicity or mock navigation
             store.setCurrentPageId('PROFILE_OVERVIEW');
             await router.push({ name: 'PROFILE_OVERVIEW' });
        }
    };

    const goBackProfile = async () => {
        store.setCurrentPageId('PROFILE_OVERVIEW');
        await router.push({ name: 'PROFILE_OVERVIEW' });
    };

    const goHome = async () => {
        store.setCurrentPageId('HOME');
        await router.push({ name: 'HOME' });
    };

    return {
        followers,
        openUser,
        goBackProfile,
        goHome
    };
  }
}
</script>