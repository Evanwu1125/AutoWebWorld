<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-10">
       <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="back-mh-list" @click="handleBack" class="mr-4 text-gray-600 hover:text-gray-900">
           <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
         </button>
         <h1 class="text-xl font-bold text-gray-900">Therapist Profile</h1>
       </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <!-- Info -->
        <div class="lg:col-span-1">
           <div class="bg-white rounded-lg shadow-lg overflow-hidden">
             <div class="relative h-64">
               <img :src="therapist?.image" :alt="therapist?.name" class="w-full h-full object-cover" />
             </div>
             <div class="p-6">
                <h2 class="text-2xl font-bold text-gray-900">{{ therapist?.name }}</h2>
                <p class="text-[#722282] font-medium mt-1">{{ therapist?.specialty }}</p>
                <div class="flex items-center mt-4">
                   <div class="bg-purple-100 text-[#722282] px-3 py-1 rounded-full text-sm font-semibold">
                     {{ therapist?.experience }} Years Exp.
                   </div>
                </div>
                <p class="mt-4 text-gray-600 text-sm">
                  Specializing in evidence-based treatments for {{ therapist?.specialty }}. Committed to creating a safe and supportive environment.
                </p>
             </div>
           </div>
        </div>

        <!-- Form -->
        <div class="lg:col-span-2 space-y-6">
          <div class="bg-white rounded-lg shadow-lg p-6">
             <h3 class="text-lg font-bold text-gray-900 mb-4">Reason for Visit</h3>
             
             <div class="mb-6">
               <label for="mh-reason-textarea" class="block text-sm font-medium text-gray-700 mb-2">
                 Please share what you'd like to discuss.
               </label>
               <textarea
                 id="mh-reason-textarea"
                 rows="5"
                 class="shadow-sm focus:ring-[#722282] focus:border-[#722282] block w-full sm:text-sm border-gray-300 rounded-md"
                 placeholder="e.g., anxiety, relationship issues, stress..."
                 @input="handleReasonInput"
               ></textarea>
             </div>

             <div class="pt-4 border-t border-gray-200">
                <button
                  id="mh-continue-schedule"
                  @click="handleContinue"
                  class="w-full bg-[#722282] text-white py-3 px-4 rounded-lg font-bold hover:bg-[#5a1a66] shadow-md transition-colors"
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
  name: 'MENTAL_HEALTH_DETAIL',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const therapist = computed(() => {
      return dataStore.therapists.find(t => t.id === store.selected_therapist_id)
    })

    const handleReasonInput = (e) => {
      // ACT_MH_DETAIL_TYPE_REASON
      store.mh_reason_for_visit = e.target.value
    }

    const handleContinue = async () => {
      // ACT_MH_DETAIL_CONTINUE
      // Precondition: mh_reason_for_visit > 0
      if (store.mh_reason_for_visit && store.mh_reason_for_visit.length > 0) {
        store.setCurrentPageId('MENTAL_HEALTH_SCHEDULE')
        await router.push({ name: 'MENTAL_HEALTH_SCHEDULE' })
      } else {
        alert('Please provide a reason for your visit.')
      }
    }

    const handleBack = async () => {
      // ACT_MH_DETAIL_BACK_LIST
      store.setCurrentPageId('MENTAL_HEALTH_LIST')
      await router.push({ name: 'MENTAL_HEALTH_LIST' })
    }

    return {
      therapist,
      handleReasonInput,
      handleContinue,
      handleBack
    }
  }
}
</script>