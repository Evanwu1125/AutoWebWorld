<template>
  <div class="h-screen flex flex-col bg-white">
    <!-- Header -->
    <div class="h-14 border-b flex items-center px-4">
      <button id="back-from-schedule" @click="handleBack" class="mr-4 text-gray-500 hover:text-gray-900">
        ← Cancel
      </button>
      <h2 class="font-bold">Schedule Message</h2>
    </div>

    <div class="flex-1 p-6 max-w-2xl mx-auto w-full">
      <!-- Text Area -->
      <div class="mb-6">
        <label class="block text-sm font-bold text-gray-700 mb-2">Message</label>
        <textarea 
            id="schedule-message-textarea"
            v-model="text"
            @input="handleType"
            class="w-full h-32 border border-gray-300 rounded-md p-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            placeholder="Type your message to schedule..."
        ></textarea>
      </div>

      <!-- Date Picker Widget -->
      <div class="mb-8">
        <label class="block text-sm font-bold text-gray-700 mb-2">Date & Time</label>
        <DateTimePicker id="date-picker" @change="handleDateChange" />
      </div>

      <!-- Submit -->
      <div class="flex justify-end">
        <button 
            id="schedule-submit-button"
            @click="handleSubmit"
            class="bg-blue-600 text-white font-bold py-2 px-6 rounded hover:bg-blue-700 disabled:opacity-50"
            :disabled="!text || !dateSet"
        >
            Schedule Message
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
  name: 'MESSAGE_SCHEDULE',
  components: { DateTimePicker },
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const text = ref('')
    const dateSet = ref(false)

    function handleType(e) {
        signatureStore.schedule_text = e.target.value
    }

    function handleDateChange(val) {
        // Val structure depends on widget, assume generic date obj or string
        signatureStore.schedule_date_time = val // '2025-10-22T10:30:00'
        dateSet.value = true
    }

    async function handleSubmit() {
        signatureStore.currentPageId = 'SCHEDULE_MESSAGE_SUCCESS'
        await router.push({ name: 'SCHEDULE_MESSAGE_SUCCESS' })
    }

    async function handleBack() {
        signatureStore.currentPageId = 'CHANNEL_DETAIL'
        await router.push({ name: 'CHANNEL_DETAIL', params: { id: signatureStore.selected_channel_id } })
    }

    return {
        signatureStore,
        text,
        dateSet,
        handleType,
        handleDateChange,
        handleSubmit,
        handleBack
    }
  }
}
</script>