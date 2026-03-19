<template>
  <div class="min-h-screen bg-gray-50 font-sans">
    <header class="bg-white shadow-sm px-4 py-3 flex items-center sticky top-0 z-20">
      <button 
        id="address-back-book" 
        class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackBook"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900 ml-2">{{ signatureStore.selected_address_id ? 'Edit Address' : 'Add New Address' }}</h1>
    </header>

    <div class="p-4 max-w-md mx-auto space-y-4">
       <div class="bg-white p-6 rounded-xl shadow-sm space-y-4">
          <div>
             <label class="block text-xs font-bold text-gray-500 uppercase mb-1">Contact Name</label>
             <input 
               id="address-full-name-input"
               type="text" 
               class="w-full border-b-2 border-gray-200 py-2 focus:border-red-500 focus:outline-none transition-colors"
               placeholder="Full Name"
               :value="signatureStore.address_full_name"
               @input="e => signatureStore.address_full_name = e.target.value"
             />
          </div>

          <div>
             <label class="block text-xs font-bold text-gray-500 uppercase mb-1">Street Address</label>
             <input 
               id="address-street-input"
               type="text" 
               class="w-full border-b-2 border-gray-200 py-2 focus:border-red-500 focus:outline-none transition-colors"
               placeholder="Street Address"
               :value="signatureStore.address_street"
               @input="e => signatureStore.address_street = e.target.value"
             />
          </div>

          <div>
             <label class="block text-xs font-bold text-gray-500 uppercase mb-1">City</label>
             <input 
               id="address-city-input"
               type="text" 
               class="w-full border-b-2 border-gray-200 py-2 focus:border-red-500 focus:outline-none transition-colors"
               placeholder="City"
               :value="signatureStore.address_city"
               @input="e => signatureStore.address_city = e.target.value"
             />
          </div>

          <div>
             <label class="block text-xs font-bold text-gray-500 uppercase mb-1">Postcode / ZIP</label>
             <input 
               id="address-postcode-input"
               type="text" 
               class="w-full border-b-2 border-gray-200 py-2 focus:border-red-500 focus:outline-none transition-colors"
               placeholder="ZIP Code"
               :value="signatureStore.address_postcode"
               @input="e => signatureStore.address_postcode = e.target.value"
             />
          </div>

          <button 
            id="address-save-button"
            class="w-full bg-red-600 text-white font-bold py-3 rounded-lg shadow-md hover:bg-red-700 transition-colors mt-4 disabled:opacity-50"
            :disabled="!canSave"
            @click="handleSave"
          >
            Save Address
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
  name: 'EDIT_ADDRESS_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const canSave = computed(() => {
       const s = signatureStore
       return s.address_full_name && s.address_street && s.address_city && s.address_postcode
    })

    const handleBackBook = async () => {
       signatureStore.currentPageId = 'ADDRESS_BOOK'
       await router.push({ name: 'ADDRESS_BOOK' })
    }

    const handleSave = async () => {
       // Save logic mock
       signatureStore.currentPageId = 'ADDRESS_BOOK'
       await router.push({ name: 'ADDRESS_BOOK' })
    }

    return {
       signatureStore,
       canSave,
       handleBackBook,
       handleSave
    }
  }
}
</script>