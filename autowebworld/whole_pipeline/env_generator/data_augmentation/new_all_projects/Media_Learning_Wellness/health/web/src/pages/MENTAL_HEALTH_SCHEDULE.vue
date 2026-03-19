<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-10">
       <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="back-mh-detail" @click="handleBack" class="mr-4 text-gray-600 hover:text-gray-900">
           <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
         </button>
         <h1 class="text-xl font-bold text-gray-900">Schedule Session</h1>
       </div>
    </header>

    <main class="flex-1 max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <div class="bg-white rounded-lg shadow-lg overflow-hidden">
        <div class="p-6 border-b border-gray-200">
           <h2 class="text-lg font-bold text-gray-900 mb-2">Select Date & Time</h2>
        </div>

        <div class="p-6 space-y-8">
           <!-- Date Picker -->
           <div>
             <label class="block text-sm font-medium text-gray-700 mb-2">Date</label>
             <div class="border rounded-md p-4 bg-gray-50">
               <DateTimePicker id="date-picker2" @change="handleDateChange" />
             </div>
           </div>

           <!-- Time Slot -->
           <div>
              <label class="block text-sm font-medium text-gray-700 mb-2">Time Slot</label>
              <div class="relative">
                 <button 
                   id="mh-time-slot-dropdown" 
                   @click="toggleTimeDropdown"
                   class="w-full bg-white border border-gray-300 rounded-md py-3 px-4 flex justify-between items-center text-left cursor-pointer focus:outline-none focus:ring-2 focus:ring-[#722282]"
                 >
                   <span class="block truncate">{{ selectedSlot || 'Select a time' }}</span>
                   <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                 </button>

                 <div v-if="timeDropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                    <div 
                      v-for="time in availableTimes" 
                      :key="time.value"
                      :id="time.selector.substring(1)"
                      @click="handleTimeSelect(time.value)"
                      class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-gray-100"
                    >
                       <span class="font-normal block truncate">{{ time.value }}</span>
                    </div>
                 </div>
              </div>
           </div>
        </div>

        <div class="p-6 bg-gray-50 border-t border-gray-200">
           <button
             id="mh-continue-review"
             @click="handleContinue"
             class="w-full bg-[#722282] text-white py-3 px-4 rounded-lg font-bold hover:bg-[#5a1a66] shadow-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
             :disabled="!selectedSlot"
           >
             Review Booking
           </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import DateTimePicker from '../components/widgets/DateTimePicker.vue'

export default {
  name: 'MENTAL_HEALTH_SCHEDULE',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const dateSelected = computed(() => store.mh_schedule_date && store.mh_schedule_date.length > 0)
    const selectedSlot = computed(() => store.mh_schedule_slot)
    
    const timeDropdownOpen = ref(false)
    const availableTimes = [
      { value: '15:00', selector: '#mh-time-slot-1500' },
      { value: '16:00', selector: '#mh-time-slot-1600' },
      { value: '17:00', selector: '#mh-time-slot-1700' }
    ]

    const handleDateChange = (date) => {
      // ACT_MH_SCHED_PICK_DATE
      store.mh_schedule_date = date
    }

    const toggleTimeDropdown = () => timeDropdownOpen.value = !timeDropdownOpen.value

    const handleTimeSelect = (time) => {
      // ACT_MH_SCHED_SELECT_SLOT
      store.mh_schedule_slot = time
      timeDropdownOpen.value = false
    }

    const handleContinue = async () => {
      // ACT_MH_SCHED_REVIEW
      if (selectedSlot.value) {
        store.setCurrentPageId('MENTAL_HEALTH_REVIEW')
        await router.push({ name: 'MENTAL_HEALTH_REVIEW' })
      }
    }

    const handleBack = async () => {
      // ACT_MH_SCHED_BACK_DETAIL
      store.setCurrentPageId('MENTAL_HEALTH_DETAIL')
      await router.push({ name: 'MENTAL_HEALTH_DETAIL' })
    }

    return {
      dateSelected,
      selectedSlot,
      timeDropdownOpen,
      availableTimes,
      handleDateChange,
      toggleTimeDropdown,
      handleTimeSelect,
      handleContinue,
      handleBack
    }
  }
}
</script>