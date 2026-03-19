<template>
  <div class="min-h-screen bg-[#FFF4E6] flex flex-col items-center py-10 px-4">
    <div class="bg-white rounded-3xl shadow-xl w-full max-w-2xl overflow-hidden border border-orange-100">
      
      <!-- Hero Image -->
      <div class="h-64 bg-gray-200 relative">
         <img v-if="session" :src="session.image" class="w-full h-full object-cover" />
         <button id="focus-detail-back-list" @click="goBack" class="absolute top-4 left-4 bg-white/90 p-2 rounded-full shadow-md hover:bg-white transition-colors text-gray-600">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
         </button>
      </div>

      <div class="p-8">
        <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ session?.title || 'Loading...' }}</h1>
        <p class="text-gray-500 mb-8 leading-relaxed">{{ session?.description }}</p>

        <!-- Volume Slider -->
        <div class="mb-8 bg-orange-50 p-6 rounded-2xl">
          <label class="block text-sm font-bold text-orange-800 mb-4 uppercase tracking-wide flex justify-between">
            <span>Music Volume</span>
            <span>{{ volume }} / 10</span>
          </label>
          <input id="focus-volume-slider" 
                 type="range" 
                 v-model.number="volume"
                 min="0" max="10" step="1"
                 @input="handleVolumeChange"
                 class="w-full h-2 bg-orange-200 rounded-lg appearance-none cursor-pointer accent-orange-500" />
        </div>

        <!-- Notes Input -->
        <div class="mb-8">
          <label class="block text-sm font-bold text-gray-700 mb-2">Focus Goal (Optional)</label>
          <textarea id="focus-notes-input" 
                    v-model="notes"
                    @input="handleNotesInput"
                    placeholder="What are you working on today?"
                    class="w-full p-4 rounded-xl border border-gray-200 focus:border-orange-500 focus:ring-orange-500 min-h-[100px] resize-none"></textarea>
        </div>

        <!-- Actions -->
        <div class="flex flex-col sm:flex-row gap-4">
          <button id="focus-start-button" 
                  @click="goToStartForm"
                  class="flex-1 bg-orange-500 hover:bg-orange-600 text-white font-bold py-4 px-6 rounded-xl shadow-lg hover:shadow-orange-500/30 transition-all transform hover:-translate-y-1">
            Start Focus Session
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
  name: 'FOCUS_DETAIL',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const sessionId = computed(() => signatureStore.selected_focus_id || route.params.id);
    const session = computed(() => dataStore.focus_sessions.find(s => s.id === sessionId.value));

    const volume = ref(7);
    const notes = ref('');

    const handleVolumeChange = () => {
      signatureStore.focus_volume_level = volume.value;
    };

    const handleNotesInput = () => {
      signatureStore.focus_notes = notes.value;
    };

    const goToStartForm = async () => {
      await router.push({ name: 'FOCUS_START_FORM' });
    };

    const goBack = async () => {
      await router.push({ name: 'FOCUS_LIST' });
    };

    onMounted(() => {
      if (sessionId.value) {
        signatureStore.selected_focus_id = sessionId.value;
      }
    });

    return {
      session,
      volume,
      notes,
      handleVolumeChange,
      handleNotesInput,
      goToStartForm,
      goBack
    };
  }
}
</script>