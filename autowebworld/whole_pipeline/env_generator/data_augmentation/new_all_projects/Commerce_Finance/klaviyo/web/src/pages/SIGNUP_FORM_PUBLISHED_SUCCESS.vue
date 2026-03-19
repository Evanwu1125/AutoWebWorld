<template>
  <div class="min-h-screen bg-slate-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full text-center space-y-8">
      <div class="mx-auto flex items-center justify-center h-24 w-24 rounded-full bg-orange-100 mb-8">
        <svg class="h-12 w-12 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
      </div>
      
      <h2 class="text-3xl font-extrabold text-slate-900">Form Published!</h2>
      <p class="text-lg text-slate-600">Your form is now collecting subscribers.</p>

      <div class="flex flex-col space-y-4 pt-8">
        <button 
          id="btn-view-forms"
          @click="viewForms"
          class="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none"
        >
          View All Forms
        </button>
        <button 
          id="btn-go-home"
          @click="goHome"
          class="w-full flex justify-center py-3 px-4 border border-slate-300 rounded-md shadow-sm text-sm font-medium text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
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
  name: 'SIGNUP_FORM_PUBLISHED_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    async function goHome() {
      store.success_message = "Signup form published"
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    async function viewForms() {
      store.setCurrentPageId('SIGNUP_FORMS_LIST')
      await router.push({ name: 'SIGNUP_FORMS_LIST' })
    }

    return {
      goHome,
      viewForms
    }
  }
}
</script>