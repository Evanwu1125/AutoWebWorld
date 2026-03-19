<template>
  <div class="min-h-screen bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-3xl mx-auto">
      <div class="text-center mb-12">
        <h1 class="text-3xl font-extrabold text-[#005DAA] sm:text-4xl">
          What kind of care do you need?
        </h1>
        <p class="mt-4 text-lg text-gray-500">
          Select the type of visit to get started.
        </p>
      </div>

      <div class="grid grid-cols-1 gap-8 sm:grid-cols-2">
        <!-- Medical Care Card -->
        <div 
          id="visit-type-medical"
          @click="handleSelectMedical"
          class="bg-white overflow-hidden shadow-lg rounded-2xl cursor-pointer hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 border-2 border-transparent hover:border-[#009CDE]"
        >
          <img src="/images/Medical.jpg" alt="General Medical" class="w-full h-48 object-cover" />
          <div class="p-6">
            <div class="flex items-center mb-4">
              <div class="bg-blue-100 rounded-full p-2 mr-3">
                <svg class="h-6 w-6 text-[#005DAA]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
              </div>
              <h3 class="text-xl font-bold text-gray-900">General Medical</h3>
            </div>
            <p class="text-gray-600">
              Cold, flu, allergies, sinus infections, and more. Talk to a doctor 24/7.
            </p>
          </div>
        </div>

        <!-- Mental Health Card -->
        <div 
          id="visit-type-mental-health"
          @click="handleSelectMentalHealth"
          class="bg-white overflow-hidden shadow-lg rounded-2xl cursor-pointer hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1 border-2 border-transparent hover:border-[#722282]"
        >
          <img src="/images/MentalHealth.jpg" alt="Mental Health" class="w-full h-48 object-cover" />
          <div class="p-6">
             <div class="flex items-center mb-4">
              <div class="bg-purple-100 rounded-full p-2 mr-3">
                <svg class="h-6 w-6 text-[#722282]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path></svg>
              </div>
              <h3 class="text-xl font-bold text-gray-900">Mental Health</h3>
            </div>
            <p class="text-gray-600">
              Therapy and psychiatry for anxiety, depression, stress, and more.
            </p>
          </div>
        </div>
      </div>

      <div class="mt-12 text-center">
        <button 
          id="back-dashboard" 
          @click="handleBackDashboard"
          class="inline-flex items-center px-6 py-3 border border-gray-300 shadow-sm text-base font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#005DAA]"
        >
          Back to Dashboard
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'VISIT_TYPE_SELECTION',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleSelectMedical = async () => {
      // ACT_VT_SELECT_MEDICAL
      // Effects: selected_visit_type = 'general_medical'
      store.selected_visit_type = 'general_medical'
      store.setCurrentPageId('PROVIDER_LIST')
      await router.push({ name: 'PROVIDER_LIST' })
    }

    const handleSelectMentalHealth = async () => {
      // ACT_VT_SELECT_MENTAL_HEALTH
      // Effects: selected_visit_type = 'mental_health'
      store.selected_visit_type = 'mental_health'
      store.setCurrentPageId('MENTAL_HEALTH_LIST')
      await router.push({ name: 'MENTAL_HEALTH_LIST' })
    }

    const handleBackDashboard = async () => {
      // ACT_VT_BACK_DASHBOARD
      store.setCurrentPageId('DASHBOARD')
      await router.push({ name: 'DASHBOARD' })
    }

    return {
      handleSelectMedical,
      handleSelectMentalHealth,
      handleBackDashboard
    }
  }
}
</script>