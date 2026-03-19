<template>
  <div class="min-h-screen bg-[#005DAA] flex flex-col items-center justify-center px-4">
    <div class="bg-white rounded-xl shadow-2xl p-8 max-w-md w-full text-center">
       <div class="mb-6">
         <div class="mx-auto flex items-center justify-center h-20 w-20 rounded-full bg-blue-100 animate-pulse">
            <svg class="h-10 w-10 text-[#005DAA]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
         </div>
       </div>
       
       <h2 class="text-2xl font-bold text-gray-900 mb-2">You are next in line</h2>
       <p class="text-gray-600 mb-8">Estimated wait time: &lt; 5 minutes</p>

       <div class="space-y-4">
          <button
            id="join-queue"
            @click="handleJoin"
            class="w-full bg-[#009CDE] text-white py-4 px-4 rounded-lg font-bold hover:bg-[#007bb0] shadow-lg transition-transform transform hover:-translate-y-1"
          >
            Start Visit Now
          </button>
          
          <button
            id="back-triage"
            @click="handleBack"
            class="w-full bg-white text-gray-500 hover:text-gray-700 font-medium py-2"
          >
            Go Back
          </button>
       </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'INSTANT_VISIT_QUEUE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleJoin = async () => {
      // ACT_QUEUE_JOIN
      // Effect: queue_position = 1
      store.queue_position = 1
      store.setCurrentPageId('INSTANT_VISIT_SUCCESS')
      await router.push({ name: 'INSTANT_VISIT_SUCCESS' })
    }

    const handleBack = async () => {
      // ACT_QUEUE_BACK_TRIAGE
      store.setCurrentPageId('INSTANT_VISIT_TRIAGE')
      await router.push({ name: 'INSTANT_VISIT_TRIAGE' })
    }

    return {
      handleJoin,
      handleBack
    }
  }
}
</script>