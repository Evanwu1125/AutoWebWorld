<template>
  <div class="min-h-screen bg-[#FFF4E6] flex flex-col items-center py-10 px-4">
    <div class="bg-white rounded-3xl shadow-xl w-full max-w-lg border border-orange-100 p-8">
      
      <div class="flex items-center gap-4 mb-8">
        <button id="focus-start-back-detail" @click="goBack" class="p-2 rounded-full hover:bg-gray-100 text-gray-500">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
            </svg>
        </button>
        <h1 class="text-2xl font-bold text-gray-800">Focus Setup</h1>
      </div>

      <!-- Task Input -->
      <div class="mb-8">
        <label class="block text-sm font-bold text-gray-700 mb-2">Main Task</label>
        <input id="focus-task-input" 
               type="text" 
               v-model="task"
               @input="handleTaskInput"
               placeholder="e.g. Finish report..." 
               class="w-full p-4 rounded-xl border border-gray-200 focus:border-orange-500 focus:ring-orange-500 bg-gray-50" />
      </div>

      <!-- Duration Select -->
      <div class="mb-8">
        <label class="block text-sm font-bold text-gray-700 mb-2">Session Length</label>
        
        <div class="relative">
          <button id="focus-duration-dropdown" 
                  @click="toggleDropdown"
                  class="w-full p-4 rounded-xl border border-gray-200 bg-white flex items-center justify-between hover:border-orange-500 transition-colors">
            <span :class="duration ? 'text-gray-800' : 'text-gray-400'">
              {{ duration ? duration : 'Select duration' }}
            </span>
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>

          <div v-if="isDropdownOpen" class="absolute top-full left-0 w-full mt-2 bg-white rounded-xl shadow-xl border border-gray-100 z-10 overflow-hidden">
            <div id="focus-duration-25" @click="selectDuration('25min')" class="p-4 hover:bg-orange-50 cursor-pointer flex items-center gap-3">
              <span>🍅</span> 25 min (Pomodoro)
            </div>
            <div id="focus-duration-45" @click="selectDuration('45min')" class="p-4 hover:bg-orange-50 cursor-pointer flex items-center gap-3">
              <span>⏱️</span> 45 min
            </div>
            <div id="focus-duration-60" @click="selectDuration('60min')" class="p-4 hover:bg-orange-50 cursor-pointer flex items-center gap-3">
              <span>⌛</span> 60 min
            </div>
          </div>
        </div>
      </div>

      <!-- Review Button -->
      <button id="focus-review-button" 
              @click="goToReview"
              :disabled="!isValid"
              class="w-full bg-orange-500 hover:bg-orange-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all">
        Review Setup
      </button>

    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'FOCUS_START_FORM',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const task = ref('');
    const duration = ref('');
    const isDropdownOpen = ref(false);

    const isValid = computed(() => {
      return task.value.length > 0 && duration.value.length > 0;
    });

    const handleTaskInput = () => {
      signatureStore.focus_task_text = task.value;
    };

    const toggleDropdown = () => isDropdownOpen.value = !isDropdownOpen.value;

    const selectDuration = (val) => {
      duration.value = val;
      signatureStore.focus_duration_choice = val;
      isDropdownOpen.value = false;
    };

    const goToReview = async () => {
      if (isValid.value) {
        await router.push({ name: 'FOCUS_REVIEW' });
      }
    };

    const goBack = async () => {
      await router.push({ name: 'FOCUS_DETAIL' });
    };

    return {
      task,
      duration,
      isDropdownOpen,
      isValid,
      handleTaskInput,
      toggleDropdown,
      selectDuration,
      goToReview,
      goBack
    };
  }
}
</script>