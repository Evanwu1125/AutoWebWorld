<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20 px-4 py-3 flex items-center justify-between">
        <div class="flex items-center">
            <button id="group-review-back-add-members" @click="goBackMembers" class="p-2 text-slate-500 hover:text-blue-600 mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
            </button>
            <h1 class="text-xl font-bold text-slate-800">Review Group</h1>
        </div>
    </header>

    <div class="flex-1 p-6 flex flex-col items-center max-w-md mx-auto w-full">
        <div class="bg-white rounded-2xl shadow-sm w-full p-6 text-center mb-6">
             <div class="w-24 h-24 bg-slate-200 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
            </div>
            <h2 class="text-2xl font-bold text-slate-900 mb-1">{{ groupName }}</h2>
            <p class="text-slate-500">{{ memberCount }} members</p>
        </div>

        <div class="bg-white rounded-2xl shadow-sm w-full p-4 flex-1 overflow-y-auto mb-6">
            <h3 class="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-4">Members</h3>
            <div class="space-y-4">
                 <div v-for="member in members" :key="member.id" class="flex items-center space-x-3">
                    <img :src="member.avatar" class="w-10 h-10 rounded-full object-cover" />
                    <span class="font-medium text-slate-800">{{ member.name }}</span>
                 </div>
            </div>
        </div>

        <button 
            id="group-create-submit" 
            @click="createGroup"
            class="w-full py-4 bg-blue-600 text-white font-bold rounded-xl hover:bg-blue-700 shadow-lg transition-transform hover:scale-[1.02]"
        >
            Create Group
        </button>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'GROUP_CREATE_REVIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const groupName = computed(() => store.group_name)
    const memberIds = computed(() => {
        if (!store.selected_member_ids) return []
        return store.selected_member_ids.map(item => item.id || item)
    })
    
    const members = computed(() => {
        return memberIds.value.map(id => dataStore.contacts.find(c => c.id === id) || { id, name: 'Unknown', avatar: '' })
    })

    const memberCount = computed(() => members.value.length)

    const goBackMembers = async () => {
        store.currentPageId = 'GROUP_CREATE_ADD_MEMBERS'
        await router.push({ name: 'GROUP_CREATE_ADD_MEMBERS' })
    }

    const createGroup = async () => {
        store.currentPageId = 'CREATE_GROUP_SUCCESS'
        await router.push({ name: 'CREATE_GROUP_SUCCESS' })
    }

    return {
        groupName,
        members,
        memberCount,
        goBackMembers,
        createGroup
    }
  }
}
</script>