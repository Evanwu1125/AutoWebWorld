<template>
  <div class="min-h-screen bg-white flex flex-col items-center justify-center p-4">
    <div class="w-full max-w-md">
       <h2 class="text-2xl font-bold font-serif text-center mb-2">Schedule Publishing</h2>
       <p class="text-gray-500 font-sans text-center mb-8">Choose a date and time to publish this story.</p>
       
       <div class="bg-white border border-gray-200 rounded-lg p-6 shadow-sm mb-8">
          <DateTimePicker id="date-picker" @change="handleDateTimeChange" />
       </div>
       
       <div class="flex flex-col gap-3">
          <button 
             id="schedule-confirm-button" 
             @click="handleConfirm" 
             :disabled="!scheduled"
             :class="{
                'w-full py-3 rounded-full font-medium font-sans transition-colors': true,
                'bg-green-600 text-white hover:bg-green-700 shadow-sm': scheduled,
                'bg-gray-200 text-gray-400 cursor-not-allowed': !scheduled
             }"
          >
             Schedule
          </button>
          
          <button 
             id="schedule-back" 
             @click="handleBack" 
             class="w-full bg-white border border-gray-300 hover:border-gray-400 text-gray-700 py-3 rounded-full font-medium font-sans transition-colors"
          >
             Cancel
          </button>
       </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import DateTimePicker from '../components/widgets/DateTimePicker.vue'

export default {
  name: 'SCHEDULE_PICKER',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    
    const scheduled = ref(false)

    const handleDateTimeChange = (isoString) => {
       signatureStore.scheduled_datetime = isoString
       scheduled.value = true
    }

    const handleConfirm = async () => {
       if (scheduled.value) {
          signatureStore.setCurrentPageId('SCHEDULE_POST_SUCCESS')
          await router.push({ name: 'SCHEDULE_POST_SUCCESS' })
       }
    }

    const handleBack = async () => {
       signatureStore.setCurrentPageId('PUBLISH_CONFIRM')
       await router.push({ name: 'PUBLISH_CONFIRM' })
    }

    return {
       scheduled,
       handleDateTimeChange,
       handleConfirm,
       handleBack
    }
  }
}
</script>