<template>
  <div class="min-h-screen bg-gray-50 pb-12">
    <!-- Back Link -->
    <div class="bg-white border-b border-gray-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div 
          id="pro-cert-back-to-list" 
          class="flex items-center text-sm text-gray-500 hover:text-gray-700 cursor-pointer w-fit"
          @click="goBack"
        >
          <svg class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
          Back to Certificates
        </div>
      </div>
    </div>

    <div v-if="cert" class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
      <div class="bg-white rounded-xl shadow-sm overflow-hidden flex flex-col lg:flex-row">
        <div class="lg:w-2/3 p-8">
          <div class="flex items-center space-x-2 text-sm text-blue-600 font-semibold uppercase tracking-wide mb-2">
            <span>{{ cert.provider }}</span>
            <span>•</span>
            <span>Professional Certificate</span>
          </div>
          
          <h1 class="text-4xl font-extrabold text-gray-900 mb-4">{{ cert.title }}</h1>
          <p class="text-xl text-gray-600 mb-6">{{ cert.description }}</p>
          
          <div class="flex items-center space-x-6 mb-8">
            <div class="flex items-center">
              <span class="text-yellow-400 text-xl mr-1">★</span>
              <span class="font-bold text-gray-900">{{ cert.rating }}</span>
            </div>
            <div class="flex items-center">
              <span class="font-bold text-gray-900">{{ cert.duration }}</span>
              <span class="text-gray-500 ml-1">months</span>
            </div>
            <div class="flex items-center">
              <span class="font-bold text-gray-900">${{ cert.price }}</span>
              <span class="text-gray-500 ml-1">one-time</span>
            </div>
          </div>
          
          <div class="prose max-w-none text-gray-600">
             <p>This professional certificate is designed to prepare you for a career in the high-growth field of {{ cert.title }}. No prior experience required.</p>
          </div>
        </div>
        
        <!-- Action Card -->
        <div class="lg:w-1/3 bg-gray-50 p-8 border-l border-gray-100 flex flex-col justify-center">
          <button 
            id="pro-cert-enroll-button"
            @click="goToEnroll"
            class="w-full bg-blue-700 hover:bg-blue-800 text-white font-bold py-4 px-6 rounded-lg shadow-md transition-colors flex justify-center items-center"
          >
            Enroll Now
          </button>
          
          <p class="text-xs text-gray-500 text-center mt-4">
             Financial aid available. 7-day money-back guarantee.
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PROFESSIONAL_CERT_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const certId = route.params.id || store.selected_pro_cert_id
    const cert = computed(() => dataStore.professional_certs.find(c => c.id === certId))

    async function goToEnroll() {
      store.setCurrentPageId('PROFESSIONAL_CERT_ENROLL_PAYMENT')
      await router.push({ name: 'PROFESSIONAL_CERT_ENROLL_PAYMENT', params: { id: certId } })
    }

    async function goBack() {
      store.setCurrentPageId('PROFESSIONAL_CERT_LIST')
      await router.push({ name: 'PROFESSIONAL_CERT_LIST' })
    }

    return {
      cert,
      goToEnroll,
      goBack
    }
  }
}
</script>