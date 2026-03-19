<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20 px-4 py-3 flex items-center justify-between">
        <div class="flex items-center">
            <button id="group-details-back" @click="goBackGroups" class="p-2 text-slate-500 hover:text-blue-600 mr-4">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
            </button>
            <h1 class="text-xl font-bold text-slate-800">New Group</h1>
        </div>
        <button 
            id="group-next-add-members" 
            @click="goNext"
            :disabled="!name.trim()"
            class="text-blue-600 font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:text-blue-700 transition-colors"
        >
            Next
        </button>
    </header>

    <div class="flex-1 p-6 flex flex-col items-center">
        <div class="w-24 h-24 bg-slate-200 rounded-full flex items-center justify-center mb-8 cursor-pointer hover:bg-slate-300 transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
        </div>

        <div class="w-full max-w-md space-y-4">
            <div>
                <label class="block text-sm font-medium text-slate-700 mb-1">Group Name</label>
                <input 
                    id="group-name-input"
                    type="text" 
                    v-model="name"
                    placeholder="Enter group name"
                    class="w-full bg-white border border-slate-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow"
                    @input="updateName"
                />
            </div>

            <div>
                <label class="block text-sm font-medium text-slate-700 mb-1">Description (Optional)</label>
                <textarea 
                    id="group-description-input"
                    v-model="description"
                    rows="3"
                    placeholder="What's this group for?"
                    class="w-full bg-white border border-slate-300 rounded-xl px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow resize-none"
                    @input="updateDescription"
                ></textarea>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'GROUP_CREATE_DETAILS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const name = ref('')
    const description = ref('')

    const updateName = () => {
        store.group_name = name.value
    }

    const updateDescription = () => {
        store.group_description = description.value
    }

    const goBackGroups = async () => {
        store.currentPageId = 'GROUPS_LIST'
        await router.push({ name: 'GROUPS_LIST' })
    }

    const goNext = async () => {
        if (!name.value.trim()) return
        store.currentPageId = 'GROUP_CREATE_ADD_MEMBERS'
        await router.push({ name: 'GROUP_CREATE_ADD_MEMBERS' })
    }

    return {
        name,
        description,
        updateName,
        updateDescription,
        goBackGroups,
        goNext
    }
  }
}
</script>