<template>
  <div class="min-h-screen bg-gray-50 font-sans pb-20">
    <header class="bg-white shadow-sm px-4 py-3 flex items-center sticky top-0 z-20">
      <button 
        id="address-back-account" 
        class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackAccount"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900 ml-2">Shipping Addresses</h1>
    </header>

    <div id="address-list-container" class="p-4 space-y-4">
       <!-- Address List -->
       <div 
         v-for="i in 3" 
         :key="i"
         :class="[
           'bg-white rounded-xl p-4 shadow-sm cursor-pointer hover:shadow-md transition-shadow border border-transparent hover:border-gray-200',
           `data-id-addr-${i}`,
           'address-row-visible'
         ]"
         @click="handleOpenAddress(`addr_${i}`)"
       >
          <div class="flex justify-between items-start mb-2">
             <span class="bg-gray-100 text-gray-600 text-xs font-bold px-2 py-0.5 rounded">Home</span>
             <button class="text-red-600 text-sm font-medium">Edit</button>
          </div>
          <h3 class="font-bold text-gray-900">John Doe</h3>
          <p class="text-sm text-gray-600">+1 234 567 890</p>
          <p class="text-sm text-gray-600 mt-1">123 Market Street, Suite 456</p>
          <p class="text-sm text-gray-600">San Francisco, CA 94105, US</p>
       </div>

       <!-- Add New Button -->
       <button 
         id="add-new-address-button"
         class="w-full border-2 border-dashed border-gray-300 rounded-xl p-4 flex items-center justify-center text-gray-500 hover:border-red-500 hover:text-red-600 hover:bg-red-50 transition-colors"
         @click="handleAddNew"
       >
         <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"></path></svg>
         Add New Address
       </button>
    </div>
  </div>
</template>

<script>
import { watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ADDRESS_BOOK',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const handleBackAccount = async () => {
       signatureStore.currentPageId = 'ACCOUNT_OVERVIEW'
       await router.push({ name: 'ACCOUNT_OVERVIEW' })
    }

    const handleOpenAddress = async (id) => {
       // Clear scroll anchor if set
       signatureStore.ADDRESS_BOOK_viewport_anchor_id = null
       signatureStore.selected_address_id = id
       signatureStore.currentPageId = 'EDIT_ADDRESS_FORM'
       await router.push({ name: 'EDIT_ADDRESS_FORM' })
    }

    const handleAddNew = async () => {
       signatureStore.currentPageId = 'EDIT_ADDRESS_FORM'
       await router.push({ name: 'EDIT_ADDRESS_FORM' })
    }

    // Scroll handler
    watch(() => signatureStore.ADDRESS_BOOK_viewport_anchor_id, async (newId) => {
      if (newId) {
        await nextTick()
        const element = document.querySelector(`.data-id-${newId}`)
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' })
        }
      }
    })

    return {
       handleBackAccount,
       handleOpenAddress,
       handleAddNew
    }
  }
}
</script>