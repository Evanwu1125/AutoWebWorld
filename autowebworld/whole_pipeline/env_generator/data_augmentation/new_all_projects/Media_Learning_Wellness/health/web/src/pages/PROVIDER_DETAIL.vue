<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-10">
       <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="back-provider-list" @click="handleBack" class="mr-4 text-gray-600 hover:text-gray-900">
           <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
         </button>
         <h1 class="text-xl font-bold text-gray-900">Provider Details</h1>
       </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <!-- Provider Info -->
        <div class="lg:col-span-1">
           <div class="bg-white rounded-lg shadow-lg overflow-hidden">
             <div class="relative h-64">
               <img :src="provider?.image" :alt="provider?.name" class="w-full h-full object-cover" />
             </div>
             <div class="p-6">
                <h2 class="text-2xl font-bold text-gray-900">{{ provider?.name }}</h2>
                <p class="text-[#005DAA] font-medium mt-1">{{ provider?.specialty }}</p>
                <div class="flex items-center mt-4">
                   <div class="bg-blue-100 text-[#005DAA] px-3 py-1 rounded-full text-sm font-semibold">
                     Rating: {{ provider?.rating }}/5.0
                   </div>
                </div>
                <p class="mt-4 text-gray-600 text-sm">
                  Dr. {{ provider?.name?.split(' ').pop() }} is a board-certified physician with extensive experience in {{ provider?.specialty }}. Dedicated to providing compassionate, high-quality care.
                </p>
             </div>
           </div>
        </div>

        <!-- Inputs Form -->
        <div class="lg:col-span-2 space-y-6">
          <div class="bg-white rounded-lg shadow-lg p-6">
             <h3 class="text-lg font-bold text-gray-900 mb-4">Reason for Visit</h3>
             
             <div class="mb-6">
               <label for="reason-for-visit-textarea" class="block text-sm font-medium text-gray-700 mb-2">
                 What is the main reason for your visit today?
               </label>
               <textarea
                 id="reason-for-visit-textarea"
                 rows="3"
                 class="shadow-sm focus:ring-[#009CDE] focus:border-[#009CDE] block w-full sm:text-sm border-gray-300 rounded-md"
                 placeholder="e.g., sore throat, follow-up, general checkup..."
                 @input="handleReasonInput"
               ></textarea>
             </div>

             <div class="mb-6">
                <label for="symptom-description-textarea" class="block text-sm font-medium text-gray-700 mb-2">
                  Please describe your symptoms in detail.
                </label>
                <textarea
                  id="symptom-description-textarea"
                  rows="4"
                  class="shadow-sm focus:ring-[#009CDE] focus:border-[#009CDE] block w-full sm:text-sm border-gray-300 rounded-md"
                  placeholder="e.g., fever for 2 days, mild cough, fatigue..."
                  @input="handleSymptomInput"
                ></textarea>
             </div>

             <div class="pt-4 border-t border-gray-200">
                <button
                  id="schedule-with-provider"
                  @click="handleSchedule"
                  class="w-full bg-[#005DAA] text-white py-3 px-4 rounded-lg font-bold hover:bg-[#004a87] shadow-md transition-colors"
                >
                  Continue to Schedule
                </button>
             </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PROVIDER_DETAIL',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const provider = computed(() => {
      return dataStore.providers.find(p => p.id === store.selected_provider_id)
    })

    const handleReasonInput = (e) => {
      // ACT_PD_TYPE_REASON
      store.selected_reason_for_visit = e.target.value
    }

    const handleSymptomInput = (e) => {
      // ACT_PD_TYPE_SYMPTOM
      store.selected_symptom_description = e.target.value
    }

    const handleSchedule = async () => {
      // ACT_PD_GO_TO_SCHEDULE
      if (store.selected_reason_for_visit && store.selected_reason_for_visit.length > 0) {
        store.setCurrentPageId('SCHEDULE_APPOINTMENT')
        await router.push({ name: 'SCHEDULE_APPOINTMENT' })
      } else {
        alert('Please enter a reason for your visit.')
      }
    }

    const handleBack = async () => {
      // ACT_PD_BACK_PROVIDERS
      store.setCurrentPageId('PROVIDER_LIST')
      await router.push({ name: 'PROVIDER_LIST' })
    }

    return {
      provider,
      handleReasonInput,
      handleSymptomInput,
      handleSchedule,
      handleBack
    }
  }
}
</script>