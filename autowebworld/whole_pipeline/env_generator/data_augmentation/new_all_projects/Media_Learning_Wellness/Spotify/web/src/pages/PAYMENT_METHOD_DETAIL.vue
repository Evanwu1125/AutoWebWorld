<template>
  <div class="flex h-screen bg-[#121212] text-white font-sans overflow-hidden">
    <main class="flex-1 overflow-y-auto p-8 md:p-12 max-w-2xl mx-auto w-full flex flex-col justify-center">
      <div id="back-payment-methods" @click="handleBackList" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8 self-start">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
         <span>Back to Methods</span>
      </div>

      <div class="bg-[#181818] p-8 rounded-xl border border-[#282828] shadow-2xl relative overflow-hidden">
         <!-- Card Visual -->
         <div class="bg-gradient-to-br from-[#1DB954] to-[#1ed760] h-48 rounded-lg p-6 text-black shadow-lg mb-8 relative">
            <div class="flex justify-between items-start">
               <div class="font-mono text-xl font-bold tracking-widest">•••• •••• •••• {{ method?.last4 }}</div>
               <div class="font-bold italic text-lg">{{ method?.type }}</div>
            </div>
            <div class="absolute bottom-6 left-6">
               <div class="text-xs uppercase opacity-75 mb-1">Card Holder</div>
               <div class="font-bold text-lg tracking-wide uppercase">{{ method?.name }}</div>
            </div>
            <div class="absolute bottom-6 right-6">
               <div class="text-xs uppercase opacity-75 mb-1">Expires</div>
               <div class="font-bold">{{ method?.expiry }}</div>
            </div>
         </div>

         <h2 class="text-2xl font-bold mb-4">Card Details</h2>
         <div class="space-y-4">
            <div class="flex justify-between border-b border-[#282828] pb-4">
               <span class="text-[#B3B3B3]">Status</span>
               <span class="font-bold" :class="method?.is_active ? 'text-[#1DB954]' : 'text-red-500'">{{ method?.is_active ? 'Active' : 'Inactive' }}</span>
            </div>
            <div class="flex justify-between border-b border-[#282828] pb-4">
               <span class="text-[#B3B3B3]">Billing Address</span>
               <span class="font-bold">Same as Profile</span>
            </div>
         </div>
         
         <div class="mt-8 flex space-x-4">
            <button class="bg-white text-black font-bold py-3 px-6 rounded-full hover:scale-105 transition-transform">Edit Method</button>
            <button class="border border-red-500 text-red-500 font-bold py-3 px-6 rounded-full hover:bg-red-500 hover:text-white transition-colors">Remove</button>
         </div>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import { useRouter, useRoute } from 'vue-router'

export default {
  name: 'PAYMENT_METHOD_DETAIL',
  setup() {
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const router = useRouter()
    const route = useRoute()

    const methodId = route.params.id || store.selected_payment_method_id
    const method = computed(() => dataStore.payment_methods.find(m => m.id === methodId))

    const handleBackList = async () => {
       store.setCurrentPageId('PAYMENT_METHODS')
       await router.push({ name: 'PAYMENT_METHODS' })
    }

    return {
       method,
       handleBackList
    }
  }
}
</script>