<template>
  <div class="h-screen flex flex-col bg-gray-50">
    <!-- Header -->
    <div class="h-14 border-b bg-white flex items-center px-4">
      <button id="back-channel-detail-from-settings" @click="handleBack" class="mr-4 text-gray-500 hover:text-gray-900">
        ← Back
      </button>
      <h2 class="font-bold">Channel Settings</h2>
    </div>

    <div class="flex-1 p-8 max-w-2xl mx-auto w-full">
      <div class="bg-white rounded-lg shadow p-6 space-y-6">
        <!-- Name -->
        <div>
            <label class="block text-sm font-bold text-gray-700 mb-1">Channel Name</label>
            <input 
                id="channel-name-input"
                type="text" 
                v-model="name"
                @input="handleTypeName"
                class="w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
                placeholder="e.g. marketing"
            >
            <p class="text-xs text-gray-500 mt-1">Names must be lowercase, without spaces or periods.</p>
        </div>

        <!-- Description -->
        <div>
            <label class="block text-sm font-bold text-gray-700 mb-1">Description</label>
            <input 
                id="channel-description-input"
                type="text" 
                v-model="description"
                @input="handleTypeDesc"
                class="w-full border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
            >
        </div>

        <!-- Privacy -->
        <div>
             <label class="block text-sm font-bold text-gray-700 mb-1">Privacy</label>
             <div class="relative">
                <div id="channel-privacy-dropdown" @click="togglePrivacy" class="w-full border border-gray-300 rounded-md p-2 bg-white cursor-pointer flex justify-between items-center">
                    <span>{{ privacy || 'Select privacy...' }}</span>
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                </div>
                <div v-if="showPrivacy" class="absolute w-full mt-1 bg-white border border-gray-200 rounded-md shadow-lg z-10">
                    <div id="channel-privacy-public" class="p-2 hover:bg-gray-100 cursor-pointer" @click="selectPrivacy('public')">Public</div>
                    <div id="channel-privacy-private" class="p-2 hover:bg-gray-100 cursor-pointer" @click="selectPrivacy('private')">Private</div>
                </div>
             </div>
        </div>

        <!-- Save -->
        <div class="pt-4 flex justify-end">
            <button 
                id="save-channel-settings" 
                @click="handleSave"
                class="bg-green-700 text-white font-bold py-2 px-6 rounded hover:bg-green-800 disabled:opacity-50"
                :disabled="!name"
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
  name: 'CHANNEL_SETTINGS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    
    const name = ref('')
    const description = ref('')
    const privacy = ref('public')
    const showPrivacy = ref(false)

    function handleTypeName(e) {
        signatureStore.channel_name = e.target.value
    }

    function handleTypeDesc(e) {
        signatureStore.channel_description = e.target.value
    }

    function togglePrivacy() {
        showPrivacy.value = !showPrivacy.value
    }

    function selectPrivacy(val) {
        privacy.value = val
        signatureStore.channel_privacy = val
        showPrivacy.value = false
    }

    async function handleSave() {
        signatureStore.currentPageId = 'CREATE_CHANNEL_SUCCESS'
        await router.push({ name: 'CREATE_CHANNEL_SUCCESS' })
    }

    async function handleBack() {
        signatureStore.currentPageId = 'CHANNEL_DETAIL'
        await router.push({ name: 'CHANNEL_DETAIL', params: { id: signatureStore.selected_channel_id } })
    }

    return {
        signatureStore,
        name,
        description,
        privacy,
        showPrivacy,
        handleTypeName,
        handleTypeDesc,
        togglePrivacy,
        selectPrivacy,
        handleSave,
        handleBack
    }
  }
}
</script>