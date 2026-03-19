<template>
  <div class="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
    <div class="sm:mx-auto sm:w-full sm:max-w-md">
      <div class="text-center">
        <div class="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-green-100">
          <svg class="h-10 w-10 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <h2 class="mt-4 text-3xl font-extrabold text-gray-900">You're Subscribed!</h2>
        <p class="mt-2 text-lg text-gray-600">{{ store.success_message }}</p>
      </div>
    </div>

    <div class="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
      <div class="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10 text-center">
        <p class="mb-6 text-gray-600">
          Your 7-day free trial has started. Happy learning!
        </p>

        <button 
          id="spec-success-go-to-learning"
          @click="goToLearning"
          class="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-700 hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 mb-4"
        >
          Go to Specialization
        </button>
        
        <div 
          id="spec-success-go-home"
          @click="goHome"
          class="text-sm font-medium text-blue-600 hover:text-blue-500 cursor-pointer"
        >
          Back to Home
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SPECIALIZATION_SUBSCRIBE_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    onMounted(() => {
      store.success_message = "Specialization subscription started"
    })

    async function goToLearning() {
      store.setCurrentPageId('SPECIALIZATION_DETAIL')
      await router.push({ name: 'SPECIALIZATION_DETAIL', params: { id: store.selected_specialization_id } })
    }

    async function goHome() {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      goToLearning,
      goHome
    }
  }
}
</script>