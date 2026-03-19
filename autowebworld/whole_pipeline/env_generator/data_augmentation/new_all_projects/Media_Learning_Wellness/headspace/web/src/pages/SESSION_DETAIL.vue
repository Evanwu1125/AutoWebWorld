<template>
  <div class="min-h-screen bg-[#FDFBF7] flex flex-col items-center py-10 px-4">
    <div class="bg-white rounded-3xl shadow-xl w-full max-w-2xl overflow-hidden border border-gray-100">
      
      <!-- Hero Image -->
      <div class="h-64 bg-gray-200 relative">
         <img v-if="session" :src="session.image" class="w-full h-full object-cover" />
         <button id="detail-back-browse" @click="goBack" class="absolute top-4 left-4 bg-white/90 p-2 rounded-full shadow-md hover:bg-white transition-colors text-gray-600">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
         </button>
      </div>

      <div class="p-8">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ session?.title || 'Loading...' }}</h1>
        <p class="text-gray-500 mb-8 leading-relaxed">{{ session?.description }}</p>

        <!-- Duration Slider -->
        <div class="mb-8 bg-orange-50 p-6 rounded-2xl">
          <label class="block text-sm font-bold text-orange-800 mb-4 uppercase tracking-wide">
            How long do you have? <span class="text-orange-600 ml-2 text-lg">{{ duration }} min</span>
          </label>
          <input id="duration-slider" 
                 type="range" 
                 v-model.number="duration"
                 min="3" max="60" step="1"
                 @input="handleDurationChange"
                 class="w-full h-2 bg-orange-200 rounded-lg appearance-none cursor-pointer accent-orange-500" />
          <div class="flex justify-between text-xs text-orange-400 mt-2 font-medium">
            <span>3 min</span>
            <span>60 min</span>
          </div>
        </div>

        <!-- Notes Input -->
        <div class="mb-8">
          <label class="block text-sm font-bold text-gray-700 mb-2">Notes before starting</label>
          <textarea id="session-notes-input" 
                    v-model="notes"
                    @input="handleNotesInput"
                    placeholder="How are you feeling right now?"
                    class="w-full p-4 rounded-xl border border-gray-200 focus:border-orange-500 focus:ring-orange-500 min-h-[100px] resize-none"></textarea>
        </div>

        <!-- Actions -->
        <div class="flex flex-col sm:flex-row gap-4">
          <button id="session-start-button" 
                  @click="goToStartForm"
                  class="flex-1 bg-orange-500 hover:bg-orange-600 text-white font-bold py-4 px-6 rounded-xl shadow-lg hover:shadow-orange-500/30 transition-all transform hover:-translate-y-1">
            Start Session
          </button>
          
          <button id="set-reminder-button" 
                  @click="goToReminder"
                  class="flex-1 bg-white hover:bg-gray-50 text-gray-700 font-bold py-4 px-6 rounded-xl border border-gray-200 shadow-sm transition-colors flex items-center justify-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Set Reminder
          </button>
        </div>

      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'SESSION_DETAIL',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const sessionId = computed(() => signatureStore.selected_session_id || route.params.id);
    const session = computed(() => dataStore.browse_sessions.find(s => s.id === sessionId.value));

    const duration = ref(10); // Default
    const notes = ref('');

    const handleDurationChange = () => {
      signatureStore.chosen_duration_minutes = duration.value;
    };

    const handleNotesInput = () => {
      signatureStore.session_notes = notes.value;
    };

    const goToStartForm = async () => {
      await router.push({ name: 'SESSION_START_FORM' });
    };

    const goToReminder = async () => {
      await router.push({ name: 'REMINDER_FORM' });
    };

    const goBack = async () => {
      await router.push({ name: 'BROWSE' });
    };

    onMounted(() => {
      // Initialize store if not set (e.g. direct nav)
      if (sessionId.value) {
        signatureStore.selected_session_id = sessionId.value;
      }
    });

    return {
      session,
      duration,
      notes,
      handleDurationChange,
      handleNotesInput,
      goToStartForm,
      goToReminder,
      goBack
    };
  }
}
</script>