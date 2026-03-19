<template>
  <div class="h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
    <div class="bg-white w-full max-w-sm rounded-2xl shadow-xl p-8 text-center space-y-6">
        <div class="w-20 h-20 bg-red-100 text-red-500 rounded-full flex items-center justify-center mx-auto">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
            </svg>
        </div>
        
        <div>
            <h2 class="text-2xl font-bold text-slate-800">User Blocked</h2>
            <p class="text-slate-500 mt-2">You will no longer receive messages from this contact.</p>
        </div>

        <div class="space-y-3 pt-4">
            <button 
                id="block-back-contacts" 
                @click="goBackContacts"
                class="w-full py-3 px-4 bg-slate-100 text-slate-700 font-semibold rounded-xl hover:bg-slate-200 transition-colors"
            >
                Back to Contacts
            </button>
            <button 
                id="block-go-home" 
                @click="goHome"
                class="w-full py-3 px-4 bg-blue-600 text-white font-semibold rounded-xl hover:bg-blue-700 shadow-md transition-colors"
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
  name: 'BLOCK_USER_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goBackContacts = async () => {
        store.currentPageId = 'CONTACTS_LIST'
        await router.push({ name: 'CONTACTS_LIST' })
    }

    const goHome = async () => {
        store.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        goBackContacts,
        goHome
    }
  }
}
</script>