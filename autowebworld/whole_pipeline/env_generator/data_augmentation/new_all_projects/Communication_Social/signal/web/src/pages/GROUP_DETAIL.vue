<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20 px-4 py-3 flex items-center">
        <button id="group-back-list" @click="goBackList" class="p-2 text-slate-500 hover:text-blue-600 mr-4">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
        </button>
        <h1 class="text-xl font-bold text-slate-800">Group Info</h1>
    </header>

    <div class="flex-1 overflow-y-auto p-4">
        <div class="max-w-md mx-auto space-y-6">
            <!-- Profile Card -->
            <div class="bg-white rounded-2xl shadow-sm p-6 flex flex-col items-center text-center">
                <img :src="group.avatar" class="w-32 h-32 rounded-full object-cover mb-4 border-4 border-slate-50" />
                <h2 class="text-2xl font-bold text-slate-900">{{ group.name }}</h2>
                <p class="text-slate-500">{{ group.member_count }} members</p>
                
                <div class="mt-6 w-full">
                    <button 
                        id="group-open-thread" 
                        @click="openThread"
                        class="w-full py-3 px-4 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 shadow-md transition-colors flex items-center justify-center space-x-2"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                            <path fill-rule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clip-rule="evenodd" />
                        </svg>
                        <span>Message Group</span>
                    </button>
                </div>
            </div>

            <!-- Members List Preview (Decorative) -->
            <div class="bg-white rounded-2xl shadow-sm p-4">
                <h3 class="text-lg font-semibold text-slate-800 mb-3">Members</h3>
                <div class="flex items-center space-x-2 overflow-x-auto pb-2">
                    <div v-for="i in 5" :key="i" class="flex-shrink-0 w-12 h-12 bg-slate-200 rounded-full flex items-center justify-center text-xs text-slate-500 font-bold border-2 border-white shadow-sm">
                        User{{i}}
                    </div>
                    <div class="flex-shrink-0 w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center text-xs text-slate-400 font-bold border-2 border-white border-dashed">
                        +{{ group.member_count - 5 }}
                    </div>
                </div>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'GROUP_DETAIL',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const groupId = computed(() => store.selected_group_id)
    const group = computed(() => dataStore.groups.find(g => g.id === groupId.value) || { name: 'Unknown Group', avatar: '/images/Group.jpg', member_count: 0 })

    const goBackList = async () => {
        store.currentPageId = 'GROUPS_LIST'
        await router.push({ name: 'GROUPS_LIST' })
    }

    const openThread = async () => {
        // Map group ID to chat ID for simplicity or create a chat entry
        // In this mock, we assume selected_chat_id can be the group ID for group chats
        store.selected_chat_id = groupId.value
        store.currentPageId = 'CHAT_THREAD'
        await router.push({ name: 'CHAT_THREAD' })
    }

    return {
        group,
        goBackList,
        openThread
    }
  }
}
</script>