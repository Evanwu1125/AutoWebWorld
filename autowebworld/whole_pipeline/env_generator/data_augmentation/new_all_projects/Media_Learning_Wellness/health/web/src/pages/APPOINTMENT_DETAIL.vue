<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-10">
       <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="back-appts-list" @click="handleBack" class="mr-4 text-gray-600 hover:text-gray-900">
           <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
         </button>
         <h1 class="text-xl font-bold text-gray-900">Appointment Details</h1>
       </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="p-6 border-b border-gray-200">
           <div class="flex items-center space-x-4">
              <img :src="appointment?.image" class="h-20 w-20 rounded-full object-cover" />
              <div>
                <h2 class="text-2xl font-bold text-gray-900">{{ appointment?.provider }}</h2>
                <p class="text-gray-500">{{ appointment?.type }} Visit</p>
              </div>
           </div>
        </div>

        <div class="p-6 space-y-6">
           <div class="grid grid-cols-2 gap-4">
              <div>
                <p class="text-sm font-medium text-gray-500">Date</p>
                <p class="text-lg text-gray-900">{{ appointment?.date }}</p>
              </div>
              <div>
                 <p class="text-sm font-medium text-gray-500">Time</p>
                 <p class="text-lg text-gray-900">{{ appointment?.time }}</p>
              </div>
           </div>

           <div class="bg-blue-50 p-4 rounded-md flex items-start space-x-3">
              <svg class="h-6 w-6 text-[#005DAA] mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              <div>
                 <h4 class="text-sm font-bold text-[#005DAA]">Instructions</h4>
                 <p class="text-sm text-blue-800 mt-1">Please arrive 15 minutes early or log in to the waiting room 5 minutes before your scheduled time.</p>
              </div>
           </div>
        </div>

        <div class="p-6 bg-gray-50 border-t border-gray-200">
           <button
             id="view-bill"
             @click="handleViewBill"
             class="w-full bg-white border border-gray-300 text-gray-700 py-3 px-4 rounded-lg font-bold hover:bg-gray-50 shadow-sm transition-colors"
           >
             View Billing
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
  name: 'APPOINTMENT_DETAIL',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const appointment = computed(() => {
      return dataStore.appointments.find(a => a.id === store.selected_appointment_id)
    })

    const handleViewBill = async () => {
      // ACT_APPT_DETAIL_GO_TO_BILLING
      // Precondition: selected_appointment_id length > 0
      if (store.selected_appointment_id) {
        store.setCurrentPageId('BILLING_OVERVIEW')
        await router.push({ name: 'BILLING_OVERVIEW' })
      }
    }

    const handleBack = async () => {
      // ACT_APPT_DETAIL_BACK_LIST
      store.setCurrentPageId('APPOINTMENTS_LIST')
      await router.push({ name: 'APPOINTMENTS_LIST' })
    }

    return {
      appointment,
      handleViewBill,
      handleBack
    }
  }
}
</script>