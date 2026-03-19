<template>
  <div class="h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
    <div class="bg-white w-full max-w-sm rounded-2xl shadow-xl p-8 text-center space-y-6">
        <div class="w-20 h-20 bg-blue-100 text-blue-500 rounded-full flex items-center justify-center mx-auto">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
        </div>
        
        <div>
            <h2 class="text-2xl font-bold text-slate-800">Group Created!</h2>
            <p class="text-slate-500 mt-2">"{{ groupName }}" is ready.</p>
        </div>

        <div class="space-y-3 pt-4">
            <button 
                id="group-success-go-groups" 
                @click="goGroups"
                class="w-full py-3 px-4 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 shadow-md transition-colors"
            >
                View Groups
            </button>
            <button 
                id="group-success-go-home" 
                @click="goHome"
                class="w-full py-3 px-4 bg-slate-100 text-slate-700 font-semibold rounded-xl hover:bg-slate-200 transition-colors"
            >
                Go Home
            </button>
        </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CREATE_GROUP_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const groupName = computed(() => store.group_name)

    const goGroups = async () => {
        store.currentPageId = 'GROUPS_LIST'
        await router.push({ name: 'GROUPS_LIST' })
    }

    const goHome = async () => {
        store.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        groupName,
        goGroups,
        goHome
    }
  }
}
</script>