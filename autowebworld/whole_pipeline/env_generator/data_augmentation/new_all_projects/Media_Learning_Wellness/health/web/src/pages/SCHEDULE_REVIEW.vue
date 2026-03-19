<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-10">
       <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="back-schedule" @click="handleBack" class="mr-4 text-gray-600 hover:text-gray-900">
           <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
         </button>
         <h1 class="text-xl font-bold text-gray-900">Review Appointment</h1>
       </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="p-6">
           <h2 class="text-xl font-bold text-gray-900 mb-6">Confirm Details</h2>
           
           <div class="space-y-6">
              <div class="flex justify-between items-start border-b border-gray-100 pb-4">
                 <div>
                    <h3 class="text-sm font-medium text-gray-500">Provider</h3>
                    <p class="mt-1 text-lg font-medium text-gray-900">{{ provider?.name }}</p>
                    <p class="text-sm text-[#005DAA]">{{ provider?.specialty }}</p>
                 </div>
                 <img :src="provider?.image" class="h-16 w-16 rounded-full object-cover" />
              </div>

              <div class="flex justify-between items-start border-b border-gray-100 pb-4">
                 <div>
                    <h3 class="text-sm font-medium text-gray-500">Date & Time</h3>
                    <p class="mt-1 text-lg font-medium text-gray-900">{{ formattedDate }}</p>
                    <p class="text-sm text-gray-600">at {{ store.schedule_selected_slot }}</p>
                 </div>
              </div>

              <div class="pb-4">
                 <h3 class="text-sm font-medium text-gray-500">Reason for Visit</h3>
                 <p class="mt-1 text-base text-gray-900">{{ store.selected_reason_for_visit }}</p>
              </div>
           </div>
        </div>

        <div class="p-6 bg-gray-50">
           <button
             id="confirm-schedule"
             @click="handleConfirm"
             class="w-full bg-[#2E7D32] text-white py-4 px-4 rounded-lg font-bold hover:bg-green-700 shadow-lg transition-transform transform hover:-translate-y-1"
           >
             Confirm Appointment
           </button>
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
  name: 'SCHEDULE_REVIEW.vue',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const provider = computed(() => {
      return dataStore.providers.find(p => p.id === store.selected_provider_id)
    })

    const formattedDate = computed(() => {
      return store.schedule_selected_date // Already in YYYY-MM-DD
    })

    const handleConfirm = async () => {
      // ACT_SCHED_REVIEW_CONFIRM
      store.setCurrentPageId('SCHEDULE_VISIT_SUCCESS')
      await router.push({ name: 'SCHEDULE_VISIT_SUCCESS' })
    }

    const handleBack = async () => {
      // ACT_SCHED_REVIEW_BACK_SCHED
      store.setCurrentPageId('SCHEDULE_APPOINTMENT')
      await router.push({ name: 'SCHEDULE_APPOINTMENT' })
    }

    return {
      store,
      provider,
      formattedDate,
      handleConfirm,
      handleBack
    }
  }
}
</script>