<template>
  <div v-if="show" class="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
    <div class="bg-white rounded-2xl shadow-2xl max-w-sm w-full overflow-hidden text-center p-6 animate-bounce-in">
      <div class="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4 text-2xl">
        📍
      </div>
      <h3 class="text-xl font-bold text-gray-900 mb-2">Location Access Needed</h3>
      <p class="text-gray-600 mb-6">
        This app needs access to your location to provide better local product recommendations and delivery estimates.
      </p>
      <button 
        id="permission-location-allow"
        @click="allow"
        class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded-xl transition-colors shadow-lg shadow-blue-200"
      >
        Allow Location Access
      </button>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'PermissionModal',
  setup() {
    const signatureStore = useSignatureStore();

    const show = computed(() => {
      // Only show on pages that require it according to FSM signature logic
      // FSM: CATEGORY_ELECTRONICS checks for global_location_permission_granted
      return signatureStore.currentPageId === 'CATEGORY_ELECTRONICS' && signatureStore.global_location_permission_granted !== true;
    });

    const allow = () => {
      signatureStore.global_location_permission_granted = true;
    };

    return {
      show,
      allow
    };
  }
}
</script>

<style scoped>
@keyframes bounceIn {
  0% { opacity: 0; transform: scale(0.9); }
  50% { opacity: 1; transform: scale(1.02); }
  100% { transform: scale(1); }
}
.animate-bounce-in {
  animation: bounceIn 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}
</style>