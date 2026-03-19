<template>
  <div class="min-h-screen bg-gray-50 pb-12">
    <!-- Back Link -->
    <div class="bg-white border-b border-gray-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div 
          id="specialization-back-to-list" 
          class="flex items-center text-sm text-gray-500 hover:text-gray-700 cursor-pointer w-fit"
          @click="goBack"
        >
          <svg class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
          Back to Specializations
        </div>
      </div>
    </div>

    <div v-if="spec" class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
      <div class="bg-white rounded-xl shadow-sm overflow-hidden flex flex-col lg:flex-row">
        <div class="lg:w-2/3 p-8">
          <div class="flex items-center space-x-2 text-sm text-blue-600 font-semibold uppercase tracking-wide mb-2">
            <span>{{ spec.university }}</span>
            <span>•</span>
            <span>Specialization</span>
          </div>
          
          <h1 class="text-4xl font-extrabold text-gray-900 mb-4">{{ spec.title }}</h1>
          <p class="text-xl text-gray-600 mb-6">{{ spec.description }}</p>
          
          <div class="flex items-center space-x-6 mb-8">
            <div class="flex items-center">
              <span class="text-yellow-400 text-xl mr-1">★</span>
              <span class="font-bold text-gray-900">{{ spec.rating }}</span>
            </div>
            <div class="flex items-center">
              <span class="font-bold text-gray-900">{{ spec.courses_count }}</span>
              <span class="text-gray-500 ml-1">courses series</span>
            </div>
            <div class="flex items-center">
              <span class="font-bold text-gray-900">{{ spec.duration }}</span>
              <span class="text-gray-500 ml-1">months</span>
            </div>
          </div>
        </div>
        
        <!-- Action Card -->
        <div class="lg:w-1/3 bg-gray-50 p-8 border-l border-gray-100 flex flex-col justify-center">
          <div class="relative w-full">
            <button 
              id="specialization-enroll-dropdown"
              @click="toggleEnrollDropdown"
              type="button" 
              class="w-full bg-blue-700 hover:bg-blue-800 text-white font-bold py-4 px-6 rounded-lg shadow-md transition-colors flex justify-between items-center"
            >
              <span>{{ selectedEnrollLabel || 'Enroll for Free' }}</span>
              <svg class="h-5 w-5 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            <div v-if="isEnrollDropdownOpen" class="absolute left-0 right-0 mt-2 bg-white rounded-md shadow-lg ring-1 ring-black ring-opacity-5 z-10">
              <div class="py-1">
                <div 
                  id="specialization-enroll-subscribe"
                  @click="selectEnrollType('subscribe', 'Start Subscription')"
                  class="block px-4 py-3 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                >
                  <div class="font-bold">Start Subscription</div>
                  <div class="text-xs text-gray-500">7-day free trial</div>
                </div>
                <div 
                  id="specialization-enroll-audit"
                  @click="selectEnrollType('audit-courses', 'Audit Courses')"
                  class="block px-4 py-3 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                >
                  <div class="font-bold">Audit Courses</div>
                  <div class="text-xs text-gray-500">Access materials only</div>
                </div>
              </div>
            </div>
          </div>

          <div v-if="store.specialization_enroll_type === 'subscribe'" class="mt-4">
            <button 
              id="specialization-continue-button"
              @click="goToSubscribePayment"
              class="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded transition-colors"
            >
              Continue
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'SPECIALIZATION_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const specId = route.params.id || store.selected_specialization_id
    const spec = computed(() => dataStore.specializations.find(s => s.id === specId))

    const isEnrollDropdownOpen = ref(false)
    const selectedEnrollLabel = ref('')

    function toggleEnrollDropdown() {
      isEnrollDropdownOpen.value = !isEnrollDropdownOpen.value
    }

    function selectEnrollType(value, label) {
      store.specialization_enroll_type = value
      selectedEnrollLabel.value = label
      isEnrollDropdownOpen.value = false
    }

    async function goToSubscribePayment() {
      if (store.specialization_enroll_type === 'subscribe') {
        store.setCurrentPageId('SPECIALIZATION_SUBSCRIBE_PAYMENT')
        await router.push({ name: 'SPECIALIZATION_SUBSCRIBE_PAYMENT', params: { id: specId } })
      }
    }

    async function goBack() {
      store.setCurrentPageId('SPECIALIZATION_LIST')
      await router.push({ name: 'SPECIALIZATION_LIST' })
    }

    return {
      store,
      spec,
      isEnrollDropdownOpen,
      selectedEnrollLabel,
      toggleEnrollDropdown,
      selectEnrollType,
      goToSubscribePayment,
      goBack
    }
  }
}
</script>