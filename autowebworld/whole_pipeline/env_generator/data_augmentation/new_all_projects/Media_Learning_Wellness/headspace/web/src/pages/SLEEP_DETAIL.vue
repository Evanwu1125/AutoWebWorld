<template>
  <div class="min-h-screen bg-[#1A2338] text-white flex flex-col items-center py-10 px-4">
    <div class="bg-[#25304C] rounded-3xl shadow-xl w-full max-w-2xl overflow-hidden border border-gray-700">
      
      <!-- Hero Image -->
      <div class="h-64 bg-gray-800 relative">
         <img v-if="track" :src="track.image" class="w-full h-full object-cover opacity-80" />
         <button id="sleep-detail-back-list" @click="goBack" class="absolute top-4 left-4 bg-black/50 backdrop-blur-md p-2 rounded-full shadow-md hover:bg-black/70 transition-colors text-white">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
         </button>
      </div>

      <div class="p-8">
        <h1 class="text-3xl font-bold text-white mb-2">{{ track?.title || 'Loading...' }}</h1>
        <p class="text-blue-200 mb-8 leading-relaxed">{{ track?.description }}</p>

        <!-- Volume Slider -->
        <div class="mb-8 bg-[#1A2338] p-6 rounded-2xl border border-gray-700">
          <label class="block text-sm font-bold text-blue-300 mb-4 uppercase tracking-wide flex justify-between">
            <span>Starting Volume</span>
            <span>{{ volume }} / 10</span>
          </label>
          <input id="sleep-volume-slider" 
                 type="range" 
                 v-model.number="volume"
                 min="0" max="10" step="1"
                 @input="handleVolumeChange"
                 class="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-500" />
        </div>

        <!-- Notes Input -->
        <div class="mb-8">
          <label class="block text-sm font-bold text-gray-300 mb-2">Sleep Journal (Optional)</label>
          <textarea id="sleep-notes-input" 
                    v-model="notes"
                    @input="handleNotesInput"
                    placeholder="Clear your mind before sleep..."
                    class="w-full p-4 rounded-xl border border-gray-600 bg-[#1A2338] text-white placeholder-gray-500 focus:border-blue-500 focus:ring-blue-500 min-h-[100px] resize-none"></textarea>
        </div>

        <!-- Actions -->
        <div class="flex flex-col sm:flex-row gap-4">
          <button id="sleep-start-button" 
                  @click="goToStartForm"
                  class="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 px-6 rounded-xl shadow-lg hover:shadow-blue-500/30 transition-all transform hover:-translate-y-1">
            Begin Sleepcast
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
  name: 'SLEEP_DETAIL',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const trackId = computed(() => signatureStore.selected_sleep_id || route.params.id);
    const track = computed(() => dataStore.sleep_tracks.find(t => t.id === trackId.value));

    const volume = ref(5);
    const notes = ref('');

    const handleVolumeChange = () => {
      signatureStore.sleep_volume_level = volume.value;
    };

    const handleNotesInput = () => {
      signatureStore.sleep_notes = notes.value;
    };

    const goToStartForm = async () => {
      await router.push({ name: 'SLEEP_START_FORM' });
    };

    const goBack = async () => {
      await router.push({ name: 'SLEEP_LIST' });
    };

    onMounted(() => {
      if (trackId.value) {
        signatureStore.selected_sleep_id = trackId.value;
      }
    });

    return {
      track,
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