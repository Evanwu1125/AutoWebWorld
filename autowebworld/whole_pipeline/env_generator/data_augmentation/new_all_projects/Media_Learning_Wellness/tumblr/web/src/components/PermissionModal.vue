<template>
  <div v-if="isOpen" class="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
    <div class="bg-slate-800 border border-slate-600 rounded-lg shadow-2xl max-w-sm w-full p-6 text-center">
      <div class="mb-4 text-4xl">📍</div>
      <h3 class="text-xl font-bold text-white mb-2">Location Access</h3>
      <p class="text-slate-300 text-sm mb-6">
        This app needs access to your location to provide better local recommendations and post tagging.
      </p>
      <div class="flex gap-3 justify-center">
        <button 
          id="permission-location-allow"
          @click="allow"
          class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-6 rounded-full transition-colors"
        >
          Allow
        </button>
        <!-- Optional Deny for UX, though FSM only specifies allow path explicitly in happy path -->
        <button 
          @click="deny"
          class="bg-transparent hover:bg-slate-700 text-slate-400 font-semibold py-2 px-6 rounded-full border border-slate-600 transition-colors"
        >
          Deny
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PermissionModal',
  setup() {
    const store = useSignatureStore()
    const route = useRoute()

    // Logic: Only show on DASHBOARD_FEED if permission is null
    // This maps to FSM logic where DASHBOARD_FEED has location_permission_granted in schema
    const isOpen = computed(() => {
      return route.name === 'DASHBOARD_FEED' && store.location_permission_granted === null
    })

    const allow = () => {
      // FSM Effect: set location_permission_granted = true
      store.location_permission_granted = true
    }
    
    const deny = () => {
      // Not in FSM happy path, but handles UI closure
      // In a real app, this might redirect or set a denied state
      alert("Location features will be disabled.")
      // For FSM compliance, we might just leave it null or handle as per app logic
      // Here we just close it visually by setting a temporary ignore or similar, 
      // but to strictly follow FSM flow, user MUST click Allow to proceed with "DASHBOARD_ALLOW_LOCATION" action.
      // So we'll encourage allowing or just do nothing on deny to force FSM path? 
      // Let's simpler: just close visually but state remains null (so it might pop up again on refresh)
    }

    return {
      isOpen,
      allow,
      deny
    }
  }
}
</script>