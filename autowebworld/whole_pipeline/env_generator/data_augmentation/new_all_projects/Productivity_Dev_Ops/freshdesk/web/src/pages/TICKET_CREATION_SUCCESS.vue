<template>
  <div class="min-h-screen bg-slate-50 flex items-center justify-center font-inter text-slate-900 px-4">
    <div class="bg-white p-8 rounded-lg shadow-lg border border-slate-200 max-w-md w-full text-center">
       <div class="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-green-100 mb-6">
          <span class="text-3xl">✅</span>
       </div>
       <h2 class="text-2xl font-bold text-slate-900 mb-2">Success!</h2>
       <p class="text-slate-600 mb-8">{{ signatureStore.success_message || 'Ticket created successfully.' }}</p>
       
       <button id="success-go-home" 
               @click="handleGoHome" 
               class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-4 rounded-md shadow-sm transition-colors duration-200">
         Go to Home
       </button>
    </div>
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'TICKET_CREATION_SUCCESS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    onMounted(() => {
        // Set success message via effect in FSM, but we can default here
        signatureStore.success_message = "Ticket created successfully"
    })

    const handleGoHome = async () => {
        signatureStore.setCurrentPageId('HOME')
        await router.push({ name: 'HOME' })
    }

    return {
        signatureStore,
        handleGoHome
    }
  }
}
</script>