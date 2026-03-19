<template>
  <div class="account-page min-h-screen bg-gray-50">
    <header class="bg-[#0071DC] text-white p-4 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center gap-4">
        <div id="account-logo-home" @click="handleGoToHome" class="font-bold text-xl cursor-pointer flex items-center gap-2">
           <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" /></svg>
           Walmart
        </div>
        <h1 class="text-lg font-medium border-l border-white/30 pl-4">My Account</h1>
      </div>
    </header>

    <main class="max-w-3xl mx-auto p-6 md:p-10">
      <div class="bg-white rounded-xl shadow-sm p-8 mb-6">
        <div class="flex items-center gap-4 mb-8">
          <div class="w-20 h-20 bg-gray-200 rounded-full overflow-hidden">
             <img src="/images/Profile.jpg" alt="Profile" class="w-full h-full object-cover" />
          </div>
          <div>
            <h2 class="text-2xl font-bold">{{ store.account_name || 'Guest User' }}</h2>
            <p class="text-gray-500">Member since 2023</p>
          </div>
        </div>

        <div class="space-y-6">
          <div class="form-group">
            <label class="block text-sm font-medium text-gray-700 mb-1">Full Name</label>
            <input 
              id="account-name-input"
              type="text" 
              :value="store.account_name"
              @input="updateName"
              placeholder="Enter your name"
              class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-shadow"
            />
          </div>
          
          <button 
            id="account-save-button"
            @click="saveName"
            class="px-6 py-2 bg-[#0071DC] text-white font-bold rounded-full hover:bg-[#005bb5] transition-colors"
          >
            Save Changes
          </button>
        </div>
      </div>

      <div class="grid md:grid-cols-2 gap-4">
        <div 
          id="account-order-history-link" 
          @click="goToOrderHistory"
          class="bg-white p-6 rounded-xl shadow-sm hover:shadow-md cursor-pointer transition-all flex items-center justify-between group"
        >
          <div class="flex items-center gap-3">
             <div class="p-3 bg-blue-50 text-blue-600 rounded-full">
               <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" /></svg>
             </div>
             <span class="font-semibold text-lg">Order History</span>
          </div>
          <svg class="w-5 h-5 text-gray-400 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" /></svg>
        </div>

        <div class="bg-white p-6 rounded-xl shadow-sm hover:shadow-md cursor-pointer transition-all flex items-center justify-between group opacity-60">
          <div class="flex items-center gap-3">
             <div class="p-3 bg-green-50 text-green-600 rounded-full">
               <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" /></svg>
             </div>
             <span class="font-semibold text-lg">Wallet</span>
          </div>
          <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" /></svg>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ACCOUNT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const updateName = (e) => {
      // FSM: ACT_ACCOUNT_EDIT_NAME
      store.account_name = e.target.value
    }

    const saveName = () => {
      // FSM: ACT_ACCOUNT_SAVE_NAME
      // Logic handled via store update already, this is just the trigger
    }

    const goToOrderHistory = async () => {
      // FSM: ACT_ACCOUNT_GO_TO_ORDER_HISTORY
      store.currentPageId = 'ORDER_HISTORY'
      await router.push({ name: 'ORDER_HISTORY' })
    }

    const handleGoToHome = async () => {
      // FSM: ACT_ACCOUNT_BACK_HOME
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      updateName,
      saveName,
      goToOrderHistory,
      handleGoToHome
    }
  }
}
</script>