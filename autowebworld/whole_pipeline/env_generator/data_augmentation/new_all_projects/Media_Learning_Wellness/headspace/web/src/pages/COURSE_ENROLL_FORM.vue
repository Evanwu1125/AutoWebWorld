<template>
  <div class="min-h-screen bg-[#FDFBF7] flex flex-col items-center py-10 px-4">
    <div class="bg-white rounded-3xl shadow-xl w-full max-w-lg border border-gray-100 p-8">
      
      <div class="flex items-center gap-4 mb-8">
        <button id="enroll-back-detail" @click="goBack" class="p-2 rounded-full hover:bg-gray-100 text-gray-500">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
            </svg>
        </button>
        <h1 class="text-2xl font-bold text-gray-800">Enrollment</h1>
      </div>

      <!-- Reason Input -->
      <div class="mb-8">
        <label class="block text-sm font-bold text-gray-700 mb-2">Why this course?</label>
        <input id="enroll-reason-input" 
               type="text" 
               v-model="reason"
               @input="handleReasonInput"
               placeholder="Your personal motivation..." 
               class="w-full p-4 rounded-xl border border-gray-200 focus:border-orange-500 focus:ring-orange-500 bg-gray-50" />
      </div>

      <!-- Frequency Select -->
      <div class="mb-8">
        <label class="block text-sm font-bold text-gray-700 mb-2">Practice Frequency</label>
        
        <div class="relative">
          <button id="enroll-frequency-dropdown" 
                  @click="toggleDropdown"
                  class="w-full p-4 rounded-xl border border-gray-200 bg-white flex items-center justify-between hover:border-orange-500 transition-colors">
            <span :class="frequency ? 'text-gray-800' : 'text-gray-400'">
              {{ frequency ? capitalize(frequency) : 'Select frequency' }}
            </span>
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>

          <div v-if="isDropdownOpen" class="absolute top-full left-0 w-full mt-2 bg-white rounded-xl shadow-xl border border-gray-100 z-10 overflow-hidden">
            <div id="enroll-frequency-daily" @click="selectFrequency('daily')" class="p-4 hover:bg-orange-50 cursor-pointer flex items-center gap-3">
              <span>📅</span> Daily
            </div>
            <div id="enroll-frequency-weekly" @click="selectFrequency('weekly')" class="p-4 hover:bg-orange-50 cursor-pointer flex items-center gap-3">
              <span>🗓️</span> Weekly
            </div>
            <div id="enroll-frequency-weekends" @click="selectFrequency('weekends')" class="p-4 hover:bg-orange-50 cursor-pointer flex items-center gap-3">
              <span>🛋️</span> Weekends
            </div>
          </div>
        </div>
      </div>

      <!-- Review Button -->
      <button id="enroll-review-button" 
              @click="goToReview"
              :disabled="!isValid"
              class="w-full bg-orange-500 hover:bg-orange-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all">
        Review Enrollment
      </button>

    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'COURSE_ENROLL_FORM',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const reason = ref('');
    const frequency = ref('');
    const isDropdownOpen = ref(false);

    const isValid = computed(() => {
      return reason.value.length > 0 && frequency.value.length > 0;
    });

    const handleReasonInput = () => {
      signatureStore.enroll_reason_text = reason.value;
    };

    const toggleDropdown = () => isDropdownOpen.value = !isDropdownOpen.value;

    const selectFrequency = (val) => {
      frequency.value = val;
      signatureStore.enroll_frequency_choice = val;
      isDropdownOpen.value = false;
    };

    const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

    const goToReview = async () => {
      if (isValid.value) {
        await router.push({ name: 'COURSE_ENROLL_REVIEW' });
      }
    };

    const goBack = async () => {
      await router.push({ name: 'COURSE_DETAIL' });
    };

    return {
      reason,
      frequency,
      isDropdownOpen,
      isValid,
      handleReasonInput,
      toggleDropdown,
      selectFrequency,
      capitalize,
      goToReview,
      goBack
    };
  }
}
</script>