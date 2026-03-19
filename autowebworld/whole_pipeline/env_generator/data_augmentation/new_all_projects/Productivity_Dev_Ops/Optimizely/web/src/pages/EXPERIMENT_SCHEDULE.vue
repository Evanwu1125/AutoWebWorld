<template>
  <div class="min-h-screen bg-gray-50 py-12">
    <div class="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="px-8 py-6 border-b border-gray-200">
          <h1 class="text-2xl font-bold text-gray-900">Schedule</h1>
          <p class="mt-1 text-sm text-gray-500">Step 4: When should this run?</p>
        </div>

        <div class="p-8 space-y-8">
          <!-- Start Date -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">Start Date & Time</label>
              <DateTimePicker @change="handleStartDate" id="date-picker1"/>
          </div>

          <!-- End Date -->
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-2">End Date & Time</label>
              <DateTimePicker @change="handleStartDate" id="date-picker2"/>
          </div>

          <!-- Launch Immediately -->
          <div class="flex items-start pt-4 border-t border-gray-200">
            <div class="flex items-center h-5">
              <input 
                id="launch-immediately-checkbox" 
                type="checkbox" 
                v-model="launchNow"
                @change="updateLaunchNow"
                class="focus:ring-blue-500 h-4 w-4 text-blue-600 border-gray-300 rounded"
              >
            </div>
            <div class="ml-3 text-sm">
              <label for="launch-immediately-checkbox" class="font-medium text-gray-700">Launch Immediately</label>
              <p class="text-gray-500">Start experiment as soon as it's saved (overrides start date).</p>
            </div>
          </div>
        </div>

        <!-- Footer -->
        <div class="bg-gray-50 px-8 py-6 flex justify-between items-center">
          <button 
            id="btn-schedule-back"
            @click="goBack"
            class="text-sm text-gray-600 hover:text-gray-900 font-medium"
          >
            Back
          </button>
          
          <div class="flex space-x-3">
            <button 
              id="btn-schedule"
              v-if="!launchNow"
              @click="schedule"
              :disabled="!isScheduleValid"
              class="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
            >
              Schedule for Later
            </button>
            <button 
              id="btn-launch-now"
              v-if="launchNow"
              @click="launch"
              :disabled="!isLaunchValid"
              class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-green-600 hover:bg-green-700 disabled:opacity-50"
            >
              Launch Now
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import DateTimePicker from '../components/widgets/DateTimePicker.vue'

export default {
  name: 'EXPERIMENT_SCHEDULE',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const launchNow = ref(false)
    
    function handleStartDate(date) {
      signatureStore.schedule_start_selected = true
    }

    function handleEndDate(date) {
      signatureStore.schedule_end_selected = true
    }

    function updateLaunchNow() {
      signatureStore.launch_immediately_checkbox = launchNow.value
    }

    const isScheduleValid = computed(() => {
      return true
    })

    const isLaunchValid = computed(() => {
      return launchNow.value
    })

    function schedule() {
      if (isScheduleValid.value) {
        signatureStore.setCurrentPageId('EXPERIMENT_SCHEDULED_SUCCESS')
        router.push({ name: 'EXPERIMENT_SCHEDULED_SUCCESS' })
      }
    }

    function launch() {
      if (isLaunchValid.value) {
        signatureStore.setCurrentPageId('EXPERIMENT_LAUNCHED_SUCCESS')
        router.push({ name: 'EXPERIMENT_LAUNCHED_SUCCESS' })
      }
    }

    function goBack() {
      signatureStore.setCurrentPageId('EXPERIMENT_EDIT_TARGETING')
      router.push({ name: 'EXPERIMENT_EDIT_TARGETING' })
    }

    return {
      launchNow,
      handleStartDate,
      handleEndDate,
      updateLaunchNow,
      isScheduleValid,
      isLaunchValid,
      schedule,
      launch,
      goBack
    }
  }
}
</script>