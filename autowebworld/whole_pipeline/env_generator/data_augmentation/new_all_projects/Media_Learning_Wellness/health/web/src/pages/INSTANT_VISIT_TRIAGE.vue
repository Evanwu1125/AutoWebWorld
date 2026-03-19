<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm z-10">
       <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
         <h1 class="text-xl font-bold text-gray-900">Instant Visit Triage</h1>
         <button id="back-home" @click="handleBackHome" class="text-gray-600 hover:text-gray-900">Cancel</button>
       </div>
    </header>

    <main class="flex-1 max-w-xl mx-auto px-4 sm:px-6 lg:px-8 py-12 w-full">
      <div class="bg-white rounded-xl shadow-lg p-8">
        <h2 class="text-2xl font-bold text-[#005DAA] mb-2">Tell us about your symptoms</h2>
        <p class="text-gray-600 mb-8">We'll connect you with the next available provider.</p>

        <div class="space-y-6">
           <div>
              <label for="instant-reason" class="block text-sm font-medium text-gray-700 mb-2">
                Reason for visit
              </label>
              <input
                id="instant-reason"
                type="text"
                class="shadow-sm focus:ring-[#009CDE] focus:border-[#009CDE] block w-full sm:text-sm border-gray-300 rounded-md py-3 px-4"
                placeholder="e.g., Fever, Sore Throat"
                @input="handleReasonInput"
              />
           </div>

           <div>
              <label for="instant-duration" class="block text-sm font-medium text-gray-700 mb-2">
                How long have you had these symptoms?
              </label>
              <input
                id="instant-duration"
                type="text"
                class="shadow-sm focus:ring-[#009CDE] focus:border-[#009CDE] block w-full sm:text-sm border-gray-300 rounded-md py-3 px-4"
                placeholder="e.g., 2 days"
                @input="handleDurationInput"
              />
           </div>

           <div class="pt-6">
              <button
                id="instant-continue"
                @click="handleContinue"
                class="w-full bg-[#005DAA] text-white py-4 px-4 rounded-lg font-bold hover:bg-[#004a87] shadow-md transition-all"
              >
                Continue to Queue
              </button>
           </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'INSTANT_VISIT_TRIAGE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleReasonInput = (e) => {
      // ACT_INSTANT_TYPE_REASON
      store.triage_reason = e.target.value
    }

    const handleDurationInput = (e) => {
      // ACT_INSTANT_TYPE_DURATION
      store.triage_symptom_duration = e.target.value
    }

    const handleContinue = async () => {
      // ACT_INSTANT_CONTINUE_QUEUE
      // Precondition: triage_reason > 0
      if (store.triage_reason && store.triage_reason.length > 0) {
        store.setCurrentPageId('INSTANT_VISIT_QUEUE')
        await router.push({ name: 'INSTANT_VISIT_QUEUE' })
      } else {
        alert('Please enter a reason for your visit.')
      }
    }

    const handleBackHome = async () => {
      // ACT_INSTANT_TRIAGE_BACK_HOME
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      handleReasonInput,
      handleDurationInput,
      handleContinue,
      handleBackHome
    }
  }
}
</script>