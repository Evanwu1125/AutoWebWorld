<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl max-w-md w-full p-8 text-center">
      <div class="mb-6 flex justify-center">
        <div class="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center text-green-500">
          <svg class="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
        </div>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Success!</h2>
      <p class="text-gray-600 mb-8">
        Your new section <span class="font-bold text-gray-900">"{{ store.new_section_name }}"</span> has been created successfully.
      </p>

      <div class="space-y-3">
        <button 
          id="section-create-success-go-sections"
          @click="goSections"
          class="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-3 rounded-lg shadow-md transition-colors"
        >
          Go to Sections
        </button>
        
        <button 
          id="section-create-success-go-home"
          @click="goHome"
          class="w-full bg-white hover:bg-gray-50 text-gray-700 font-bold py-3 rounded-lg border border-gray-200 transition-colors"
        >
          Return Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SECTION_CREATE_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goSections = async () => {
      store.setCurrentPageId('SECTION_LIST')
      await router.push({ name: 'SECTION_LIST' })
    }

    const goHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      goSections,
      goHome
    }
  }
}
</script>