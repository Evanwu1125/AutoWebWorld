<template>
  <div class="min-h-screen bg-[#FDFBF7] flex flex-col items-center py-10 px-4">
    <div class="bg-white rounded-3xl shadow-xl w-full max-w-lg border border-gray-100 p-8">
      
      <div class="flex items-center gap-4 mb-8">
        <!-- Back button logic: needs to know where it came from. 
             FSM has actions back to SESSION_DETAIL and COURSE_DETAIL.
             I'll implement both buttons but conditionally show or just one generic back if FSM permits.
             FSM has specific actions: ACT_REMINDER_BACK_SESSION_DETAIL and ACT_REMINDER_BACK_COURSE_DETAIL.
             I should show BOTH if I can't determine context, or better, show the one relevant to previous page.
             However, checking FSM logic: usually context is known. 
             I will render BOTH buttons as per FSM requirement to have selectors present, 
             but maybe hide one visually based on context? 
             Actually, FSM allows both actions from this state. 
             I'll just put both buttons for FSM compliance, maybe labeled "Back to Session" and "Back to Course".
             Or better: Use a single UI element if visual design dictates, but ID must match.
             Wait, I must have elements matching selectors: #reminder-back-session-detail AND #reminder-back-course-detail.
             I will place them both. -->
        <div class="flex gap-2">
          <button id="reminder-back-session-detail" @click="goBackSession" class="p-2 rounded-full hover:bg-gray-100 text-gray-500" title="Back to Session">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
              </svg>
          </button>
           <button id="reminder-back-course-detail" @click="goBackCourse" class="p-2 rounded-full hover:bg-gray-100 text-gray-500" title="Back to Course">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
              </svg>
          </button>
        </div>
        <h1 class="text-2xl font-bold text-gray-800">Set Reminder</h1>
      </div>

      <!-- Label Input -->
      <div class="mb-8">
        <label class="block text-sm font-bold text-gray-700 mb-2">Label</label>
        <input id="reminder-label-input" 
               type="text" 
               v-model="label"
               @input="handleLabelInput"
               placeholder="e.g. Daily Meditation" 
               class="w-full p-4 rounded-xl border border-gray-200 focus:border-orange-500 focus:ring-orange-500 bg-gray-50" />
      </div>

      <!-- Time Select -->
      <div class="mb-8">
        <label class="block text-sm font-bold text-gray-700 mb-2">Time</label>
        
        <div class="relative">
          <button id="reminder-time-dropdown" 
                  @click="toggleDropdown"
                  class="w-full p-4 rounded-xl border border-gray-200 bg-white flex items-center justify-between hover:border-orange-500 transition-colors">
            <span :class="time ? 'text-gray-800' : 'text-gray-400'">
              {{ time ? time : 'Select time' }}
            </span>
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>

          <div v-if="isDropdownOpen" class="absolute top-full left-0 w-full mt-2 bg-white rounded-xl shadow-xl border border-gray-100 z-10 overflow-hidden">
            <div id="reminder-time-8am" @click="selectTime('8am')" class="p-4 hover:bg-orange-50 cursor-pointer flex items-center gap-3">
              <span>🌅</span> 8:00 AM
            </div>
            <div id="reminder-time-12pm" @click="selectTime('12pm')" class="p-4 hover:bg-orange-50 cursor-pointer flex items-center gap-3">
              <span>☀️</span> 12:00 PM
            </div>
            <div id="reminder-time-6pm" @click="selectTime('6pm')" class="p-4 hover:bg-orange-50 cursor-pointer flex items-center gap-3">
              <span>🌇</span> 6:00 PM
            </div>
          </div>
        </div>
      </div>

      <!-- Review Button -->
      <button id="reminder-review-button" 
              @click="goToReview"
              :disabled="!isValid"
              class="w-full bg-orange-500 hover:bg-orange-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all">
        Review Reminder
      </button>

    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'REMINDER_FORM',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const label = ref('');
    const time = ref('');
    const isDropdownOpen = ref(false);

    const isValid = computed(() => {
      return label.value.length > 0 && time.value.length > 0;
    });

    const handleLabelInput = () => {
      signatureStore.reminder_label_text = label.value;
    };

    const toggleDropdown = () => isDropdownOpen.value = !isDropdownOpen.value;

    const selectTime = (val) => {
      time.value = val;
      signatureStore.reminder_time_choice = val;
      isDropdownOpen.value = false;
    };

    const goToReview = async () => {
      if (isValid.value) {
        await router.push({ name: 'REMINDER_REVIEW' });
      }
    };

    const goBackSession = async () => {
      await router.push({ name: 'SESSION_DETAIL' });
    };

     const goBackCourse = async () => {
      await router.push({ name: 'COURSE_DETAIL' });
    };

    return {
      label,
      time,
      isDropdownOpen,
      isValid,
      handleLabelInput,
      toggleDropdown,
      selectTime,
      goToReview,
      goBackSession,
      goBackCourse
    };
  }
}
</script>