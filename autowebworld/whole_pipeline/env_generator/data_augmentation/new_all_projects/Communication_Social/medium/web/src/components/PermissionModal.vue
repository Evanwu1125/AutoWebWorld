<template>
  <div v-if="visible" class="fixed inset-0 z-[9999] flex items-center justify-center bg-black/50 backdrop-blur-sm">
    <div class="bg-white rounded-xl shadow-2xl p-8 max-w-sm w-full mx-4 text-center animate-scale-in">
      <div class="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
        </svg>
      </div>
      
      <h3 class="text-xl font-bold text-gray-900 mb-3">Location Permission Required</h3>
      <p class="text-gray-600 mb-8 leading-relaxed">
        This app needs access to your location to provide better service and localized content.
      </p>
      
      <div class="flex flex-col space-y-3">
        <button
          id="permission-location-allow"
          class="w-full bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-full font-medium transition-colors shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
          @click="allowPermission"
        >
          Allow Location Access
        </button>
        <button
          id="permission-location-deny"
          class="w-full text-gray-500 hover:text-gray-800 px-6 py-2 text-sm font-medium transition-colors"
          @click="denyPermission"
        >
          Maybe Later
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, watch, onMounted } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useRoute } from 'vue-router'

export default {
  name: 'PermissionModal',
  setup() {
    const signatureStore = useSignatureStore()
    const route = useRoute()
    const visible = ref(false)

    // List of pages requiring permission (checked from FSM)
    const permissionPages = ['POST_LIST'] // Only POST_LIST has location_permission_granted in schema

    const checkPermission = () => {
      if (permissionPages.includes(route.name) && signatureStore.location_permission_granted !== true) {
        visible.value = true
      } else {
        visible.value = false
      }
    }

    watch(() => route.name, () => {
      checkPermission()
    })
    
    watch(() => signatureStore.location_permission_granted, (newVal) => {
      if (newVal === true) {
        visible.value = false
      }
    })

    onMounted(() => {
      checkPermission()
    })

    const allowPermission = () => {
      signatureStore.location_permission_granted = true
      visible.value = false
    }

    const denyPermission = () => {
      visible.value = false
      // Optionally handle denial state
    }

    return {
      visible,
      allowPermission,
      denyPermission
    }
  }
}
</script>

<style scoped>
.animate-scale-in {
  animation: scaleIn 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes scaleIn {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}
</style>