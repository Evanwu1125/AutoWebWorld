<template>
  <div class="h-screen flex flex-col bg-gray-50 items-center justify-center p-4">
    <div class="bg-white rounded-lg shadow-lg w-full max-w-md overflow-hidden">
        <div class="h-14 border-b flex items-center justify-between px-4 bg-gray-50">
            <h2 class="font-bold">Edit Profile</h2>
            <button id="back-profile-view" @click="handleBack" class="text-gray-500 hover:text-gray-900">Cancel</button>
        </div>

        <div class="p-6 space-y-4">
            <!-- Name -->
            <div>
                <label class="block text-sm font-bold text-gray-700 mb-1">Full Name</label>
                <input 
                    id="profile-name-input"
                    type="text" 
                    v-model="name"
                    @input="handleTypeName"
                    class="w-full border-gray-300 rounded-md p-2"
                >
            </div>

            <!-- Title -->
            <div>
                <label class="block text-sm font-bold text-gray-700 mb-1">Job Title</label>
                <input 
                    id="profile-title-input"
                    type="text" 
                    v-model="title"
                    @input="handleTypeTitle"
                    class="w-full border-gray-300 rounded-md p-2"
                >
            </div>

            <!-- Status Dropdown -->
            <div>
                <label class="block text-sm font-bold text-gray-700 mb-1">Status</label>
                <div class="relative">
                    <div id="status-dropdown" @click="toggleStatus" class="w-full border border-gray-300 rounded-md p-2 bg-white cursor-pointer flex justify-between items-center">
                        <span>{{ status || 'Select status...' }}</span>
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                    </div>
                    <div v-if="showStatus" class="absolute w-full mt-1 bg-white border border-gray-200 rounded-md shadow-lg z-10">
                        <div id="status-available" class="p-2 hover:bg-gray-100 cursor-pointer" @click="selectStatus('available')">Available</div>
                        <div id="status-away" class="p-2 hover:bg-gray-100 cursor-pointer" @click="selectStatus('away')">Away</div>
                        <div id="status-in-meeting" class="p-2 hover:bg-gray-100 cursor-pointer" @click="selectStatus('in_meeting')">In a meeting</div>
                    </div>
                </div>
            </div>

            <!-- Save -->
            <div class="pt-4">
                <button 
                    id="save-profile-button"
                    @click="handleSave"
                    class="w-full bg-blue-600 text-white font-bold py-2 rounded hover:bg-blue-700"
                >
                    Save Changes
                </button>
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
  name: 'PROFILE_EDIT',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    
    const name = ref('')
    const title = ref('')
    const status = ref('available')
    const showStatus = ref(false)

    function handleTypeName(e) {
        signatureStore.profile_name = e.target.value
    }

    function handleTypeTitle(e) {
        signatureStore.profile_title = e.target.value
    }

    function toggleStatus() {
        showStatus.value = !showStatus.value
    }

    function selectStatus(val) {
        status.value = val
        signatureStore.profile_status = val
        showStatus.value = false
    }

    async function handleSave() {
        signatureStore.currentPageId = 'UPDATE_PROFILE_SUCCESS'
        await router.push({ name: 'UPDATE_PROFILE_SUCCESS' })
    }

    async function handleBack() {
        signatureStore.currentPageId = 'PROFILE_VIEW'
        await router.push({ name: 'PROFILE_VIEW' })
    }

    return {
        signatureStore,
        name,
        title,
        status,
        showStatus,
        handleTypeName,
        handleTypeTitle,
        toggleStatus,
        selectStatus,
        handleSave,
        handleBack
    }
  }
}
</script>