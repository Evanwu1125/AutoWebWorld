<template>
  <div class="h-screen flex flex-col bg-gray-50">
    <!-- Header -->
    <header class="bg-[#6264A7] text-white p-4 shadow-md flex justify-between items-center z-20">
      <div class="font-bold text-lg flex items-center">
        <button id="meeting-details-back" @click="goBack" class="mr-4 hover:bg-[#464775] p-1 rounded">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        Schedule Meeting
      </div>
    </header>

    <main class="flex-1 flex flex-col p-8 items-center justify-center overflow-y-auto">
      <div class="bg-white rounded-lg shadow-lg p-8 w-full max-w-3xl border border-gray-200">
        <h2 class="text-2xl font-bold text-gray-800 mb-6 border-b pb-4">New Meeting Details</h2>
        
        <div class="space-y-6">
          <!-- Title Input ACT_MEETING_DETAILS_TYPE_TITLE -->
          <div>
            <label class="block text-sm font-semibold text-gray-700 mb-1">Title</label>
            <input 
              id="meeting-title-input"
              type="text" 
              v-model="title"
              placeholder="Add title"
              class="w-full rounded-md border-gray-300 shadow-sm focus:border-[#6264A7] focus:ring-[#6264A7] px-4 py-2 border text-lg"
            />
          </div>

          <!-- Description Input ACT_MEETING_DETAILS_TYPE_DESCRIPTION -->
          <div>
            <label class="block text-sm font-semibold text-gray-700 mb-1">Description</label>
            <textarea 
              id="meeting-description-input"
              v-model="description"
              rows="4"
              placeholder="Type details for this new meeting"
              class="w-full rounded-md border-gray-300 shadow-sm focus:border-[#6264A7] focus:ring-[#6264A7] px-4 py-2 border resize-none"
            ></textarea>
          </div>

          <!-- Date/Time Picker ACT_MEETING_DETAILS_SELECT_DATETIME -->
          <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
             <div>
                <label class="block text-sm font-semibold text-gray-700 mb-1">Date & Time</label>
                <DateTimePicker
                  id="date-picker"
                  v-model="dateTime"
                  class="w-full border rounded-md"
                />
             </div>
             <div class="flex items-center justify-center bg-gray-50 rounded-lg p-4 text-center">
                <div>
                    <p class="text-gray-500 text-sm mb-1">Selected:</p>
                    <p class="font-semibold text-lg text-[#6264A7]">{{ dateTime || 'No date/time selected' }}</p>
                </div>
             </div>
          </div>
        </div>

        <div class="mt-8 flex justify-end pt-6 border-t border-gray-100">
          <button 
            id="meeting-next-button"
            @click="nextStep"
            :disabled="!isValid"
            class="bg-[#6264A7] hover:bg-[#464775] text-white font-semibold py-2 px-8 rounded shadow-sm disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            Next: Review
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import DateTimePicker from '../components/widgets/DateTimePicker.vue'

export default {
  name: 'MEETING_DETAILS',
  components: { DateTimePicker },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const title = ref('')
    const description = ref('')
    const dateTime = ref('')

    const isValid = computed(() => {
      return title.value.trim().length > 0 && dateTime.value.trim().length > 0
    })

    // Watch for changes and sync to store
    watch(title, (val) => {
      store.meeting_title = val
    })

    watch(description, (val) => {
      store.meeting_description = val
    })

    watch(dateTime, (val) => {
      // Parse 'YYYY-MM-DD HH:mm' format
      if (val && val.includes(' ')) {
        const [datePart, timePart] = val.split(' ')
        store.meeting_date = datePart
        store.meeting_time = timePart
      }
    })

    const nextStep = async () => {
      if (!isValid.value) return;

      store.currentPageId = 'MEETING_REVIEW';
      await router.push({ name: 'MEETING_REVIEW' });
    }

    const goBack = async () => {
      store.currentPageId = 'CALENDAR_VIEW';
      await router.push({ name: 'CALENDAR_VIEW' });
    }

    return {
      title,
      description,
      dateTime,
      isValid,
      nextStep,
      goBack
    }
  }
}
</script>