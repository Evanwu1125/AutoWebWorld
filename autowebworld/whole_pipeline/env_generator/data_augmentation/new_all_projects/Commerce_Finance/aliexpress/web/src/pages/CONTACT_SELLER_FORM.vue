<template>
  <div class="min-h-screen bg-gray-50 font-sans">
    <header class="bg-white shadow-sm px-4 py-3 flex items-center sticky top-0 z-20">
      <button 
        id="contact-back-product" 
        class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackProduct"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900 ml-2">Contact Seller</h1>
    </header>

    <div class="p-4 max-w-md mx-auto space-y-4">
       <div class="bg-white p-6 rounded-xl shadow-sm space-y-4">
          <div>
             <label class="block text-xs font-bold text-gray-500 uppercase mb-1">Subject</label>
             <input 
               id="contact-subject-input"
               type="text" 
               class="w-full border-b-2 border-gray-200 py-2 focus:border-red-500 focus:outline-none transition-colors"
               placeholder="What is this about?"
               :value="signatureStore.message_subject"
               @input="e => signatureStore.message_subject = e.target.value"
             />
          </div>

          <div>
             <label class="block text-xs font-bold text-gray-500 uppercase mb-1">Message</label>
             <textarea 
               id="contact-body-textarea"
               rows="4"
               class="w-full border-2 border-gray-200 rounded-lg p-3 focus:border-red-500 focus:outline-none transition-colors text-sm"
               placeholder="Type your question here..."
               :value="signatureStore.message_body"
               @input="e => signatureStore.message_body = e.target.value"
             ></textarea>
          </div>

          <button 
            id="contact-submit-button"
            class="w-full bg-red-600 text-white font-bold py-3 rounded-lg shadow-md hover:bg-red-700 transition-colors mt-2 disabled:opacity-50"
            :disabled="!canSubmit"
            @click="handleSubmit"
          >
            Send Message
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
  name: 'CONTACT_SELLER_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const canSubmit = computed(() => {
       const s = signatureStore
       return s.message_subject && s.message_body
    })

    const handleBackProduct = async () => {
       signatureStore.currentPageId = 'PRODUCT_DETAIL'
       await router.push({ name: 'PRODUCT_DETAIL' })
    }

    const handleSubmit = async () => {
       signatureStore.success_message = 'Message Sent!'
       signatureStore.currentPageId = 'CONTACT_SELLER_SUCCESS'
       await router.push({ name: 'CONTACT_SELLER_SUCCESS' })
    }

    return {
       signatureStore,
       canSubmit,
       handleBackProduct,
       handleSubmit
    }
  }
}
</script>