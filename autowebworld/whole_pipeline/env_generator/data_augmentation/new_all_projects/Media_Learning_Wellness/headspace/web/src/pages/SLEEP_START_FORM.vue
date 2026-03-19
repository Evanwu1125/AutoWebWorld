<template>
  <div class="min-h-screen bg-[#1A2338] text-white flex flex-col items-center py-10 px-4">
    <div class="bg-[#25304C] rounded-3xl shadow-xl w-full max-w-lg border border-gray-700 p-8">
      
      <div class="flex items-center gap-4 mb-8">
        <button id="sleep-start-back-detail" @click="goBack" class="p-2 rounded-full hover:bg-[#324164] text-gray-400">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
            </svg>
        </button>
        <h1 class="text-2xl font-bold text-white">Wind Down Setup</h1>
      </div>

      <!-- Bedtime Input -->
      <div class="mb-8">
        <label class="block text-sm font-bold text-gray-300 mb-2">Ideal Bedtime</label>
        <input id="sleep-bedtime-input" 
               type="text" 
               v-model="bedtime"
               @input="handleBedtimeInput"
               placeholder="e.g. 10:30 PM" 
               class="w-full p-4 rounded-xl border border-gray-600 bg-[#1A2338] text-white placeholder-gray-500 focus:border-blue-500 focus:ring-blue-500" />
      </div>

      <!-- Environment Select -->
      <div class="mb-8">
        <label class="block text-sm font-bold text-gray-300 mb-2">Lighting Preference</label>
        
        <div class="relative">
          <button id="sleep-environment-dropdown" 
                  @click="toggleDropdown"
                  class="w-full p-4 rounded-xl border border-gray-600 bg-[#1A2338] flex items-center justify-between hover:border-blue-500 transition-colors">
            <span :class="environment ? 'text-white' : 'text-gray-500'">
              {{ environment ? capitalize(environment) : 'Select lighting' }}
            </span>
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-500" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>

          <div v-if="isDropdownOpen" class="absolute top-full left-0 w-full mt-2 bg-[#1A2338] rounded-xl shadow-xl border border-gray-700 z-10 overflow-hidden">
            <div id="sleep-environment-dark" @click="selectEnvironment('dark')" class="p-4 hover:bg-[#324164] cursor-pointer flex items-center gap-3 text-white">
              <span>🌑</span> Pitch Dark
            </div>
            <div id="sleep-environment-dim" @click="selectEnvironment('dim')" class="p-4 hover:bg-[#324164] cursor-pointer flex items-center gap-3 text-white">
              <span>🕯️</span> Dim Light
            </div>
            <div id="sleep-environment-light" @click="selectEnvironment('light')" class="p-4 hover:bg-[#324164] cursor-pointer flex items-center gap-3 text-white">
              <span>💡</span> Night Light
            </div>
          </div>
        </div>
      </div>

      <!-- Review Button -->
      <button id="sleep-review-button" 
              @click="goToReview"
              :disabled="!isValid"
              class="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed disabled:text-gray-400 text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all">
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
  name: 'SLEEP_START_FORM',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const bedtime = ref('');
    const environment = ref('');
    const isDropdownOpen = ref(false);

    const isValid = computed(() => {
      return bedtime.value.length > 0 && environment.value.length > 0;
    });

    const handleBedtimeInput = () => {
      signatureStore.sleep_bedtime_text = bedtime.value;
    };

    const toggleDropdown = () => isDropdownOpen.value = !isDropdownOpen.value;

    const selectEnvironment = (val) => {
      environment.value = val;
      signatureStore.sleep_environment_choice = val;
      isDropdownOpen.value = false;
    };

    const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

    const goToReview = async () => {
      if (isValid.value) {
        await router.push({ name: 'SLEEP_REVIEW' });
      }
    };

    const goBack = async () => {
      await router.push({ name: 'SLEEP_DETAIL' });
    };

    return {
      bedtime,
      environment,
      isDropdownOpen,
      isValid,
      handleBedtimeInput,
      toggleDropdown,
      selectEnvironment,
      capitalize,
      goToReview,
      goBack
    };
  }
}
</script>