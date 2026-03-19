<template>
  <div class="min-h-screen bg-white flex flex-col items-center justify-center p-4">
    <div class="w-full max-w-md bg-white rounded-xl shadow-2xl border border-gray-100 p-8 text-center">
      <h2 class="text-2xl font-bold mb-2 text-gray-800">Schedule Post</h2>
      <p class="text-gray-500 mb-8 text-sm">Pick a time in the future.</p>

      <!-- Date Picker Widget -->
      <div id="date-picker" class="mb-8 p-4 bg-gray-50 rounded-lg border border-gray-200">
         <DateTimePicker 
            :model-value="dateValue"
            @update:model-value="handleDateChange"
         />
      </div>

      <!-- Actions -->
      <div class="flex gap-4">
        <button 
          id="schedule-back-compose" 
          @click="goBack"
          class="flex-1 py-3 px-4 rounded-full font-bold text-gray-500 hover:bg-gray-100 transition-colors"
        >
          Cancel
        </button>
        <button 
          id="schedule-submit-button" 
          @click="submitSchedule"
          :disabled="!isValid"
          :class="[
            'flex-1 py-3 px-4 rounded-full font-bold text-white transition-all transform',
            isValid ? 'bg-blue-500 hover:bg-blue-600 hover:scale-105 shadow-lg shadow-blue-500/30' : 'bg-gray-300 cursor-not-allowed'
          ]"
        >
          Schedule
        </button>
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
  name: 'SCHEDULE_POST',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    // Internal state for picker
    const dateValue = ref(store.schedule_datetime || '')

    const handleDateChange = (newVal) => {
      dateValue.value = newVal
      store.schedule_datetime = newVal
    }

    const isValid = computed(() => {
      return (store.compose_body?.length > 0) && (store.schedule_datetime?.length > 0)
    })

    const goBack = async () => {
      store.currentPageId = 'COMPOSE_TEXT_POST'
      await router.push({ name: 'COMPOSE_TEXT_POST' })
    }

    const submitSchedule = async () => {
      if (!isValid.value) return
      store.success_message = `Scheduled for ${new Date(store.schedule_datetime).toLocaleString()}`
      store.currentPageId = 'POST_SCHEDULE_SUCCESS'
      await router.push({ name: 'POST_SCHEDULE_SUCCESS' })
    }

    return {
      store,
      dateValue,
      handleDateChange,
      isValid,
      goBack,
      submitSchedule
    }
  }
}
</script>