<template>
  <div class="h-screen flex flex-col items-center justify-center bg-indigo-50">
    <div class="bg-white p-8 rounded-xl shadow-lg text-center max-w-sm w-full">
      <div class="w-16 h-16 bg-indigo-100 text-indigo-600 rounded-full flex items-center justify-center mx-auto mb-4">
        <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
      </div>
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Profile Updated!</h2>
      <p class="text-gray-600 mb-6">{{ signatureStore.success_message || 'Your profile changes have been saved.' }}</p>
      
      <div class="space-y-3">
        <button 
            id="back-to-profile-view-from-success" 
            @click="handleBackProfile"
            class="w-full bg-indigo-600 text-white font-bold py-2 rounded hover:bg-indigo-700"
        >
            View Profile
        </button>
        <button 
            id="go-home-from-profile-success" 
            @click="handleGoHome"
            class="w-full text-gray-600 hover:text-gray-900 font-medium"
        >
            Go Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'UPDATE_PROFILE_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    async function handleBackProfile() {
        signatureStore.currentPageId = 'PROFILE_VIEW'
        await router.push({ name: 'PROFILE_VIEW' })
    }

    async function handleGoHome() {
        signatureStore.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        signatureStore,
        handleBackProfile,
        handleGoHome
    }
  }
}
</script>