<template>
  <div class="h-screen flex flex-col bg-gray-50 items-center justify-center p-4">
    <div class="bg-white rounded-lg shadow-lg w-full max-w-md overflow-hidden">
      <!-- Header / Cover -->
      <div class="h-24 bg-purple-900 relative">
        <button id="back-workspace-from-profile" @click="handleBackWS" class="absolute top-4 left-4 text-white hover:underline text-sm bg-black/20 px-2 py-1 rounded">
            Back to Workspace
        </button>
        <button id="back-to-channels-from-profile" @click="handleBackChannels" class="absolute top-4 right-4 text-white hover:underline text-sm bg-black/20 px-2 py-1 rounded">
            Back to Channels
        </button>
      </div>
      
      <!-- Profile Content -->
      <div class="px-6 pb-6 relative">
        <div class="w-24 h-24 rounded-lg bg-gray-300 border-4 border-white absolute -top-12 overflow-hidden shadow-sm">
             <img src="/images/photo1765160887.jpg" class="w-full h-full object-cover" />
        </div>
        
        <div class="mt-14 flex justify-between items-start">
            <div>
                <h2 class="text-2xl font-bold text-gray-900">You</h2>
                <p class="text-gray-600">Product Designer</p>
            </div>
            <button 
                id="edit-profile-button" 
                @click="handleEdit"
                class="border border-gray-300 text-gray-700 px-4 py-1 rounded font-semibold hover:bg-gray-50 transition"
            >
                Edit Profile
            </button>
        </div>

        <div class="mt-6 space-y-4">
            <div class="flex items-center text-gray-700">
                <span class="w-24 text-gray-500 text-sm">Status</span>
                <span class="flex items-center">
                    <span class="w-2 h-2 bg-green-500 rounded-full mr-2"></span> Available
                </span>
            </div>
            <div class="flex items-center text-gray-700">
                <span class="w-24 text-gray-500 text-sm">Local Time</span>
                <span>10:42 AM</span>
            </div>
             <div class="flex items-center text-gray-700">
                <span class="w-24 text-gray-500 text-sm">Email</span>
                <a href="#" class="text-blue-600 hover:underline">user@acme.com</a>
            </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PROFILE_VIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    async function handleEdit() {
        signatureStore.currentPageId = 'PROFILE_EDIT'
        await router.push({ name: 'PROFILE_EDIT' })
    }

    async function handleBackWS() {
        signatureStore.currentPageId = 'WORKSPACE_OVERVIEW'
        await router.push({ name: 'WORKSPACE_OVERVIEW' })
    }

    async function handleBackChannels() {
        signatureStore.currentPageId = 'CHANNEL_LIST'
        await router.push({ name: 'CHANNEL_LIST' })
    }

    return {
        signatureStore,
        handleEdit,
        handleBackWS,
        handleBackChannels
    }
  }
}
</script>