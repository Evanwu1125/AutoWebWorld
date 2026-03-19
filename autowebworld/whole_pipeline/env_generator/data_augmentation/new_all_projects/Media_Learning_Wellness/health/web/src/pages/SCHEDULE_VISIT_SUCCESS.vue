<template>
  <div class="min-h-screen bg-white flex flex-col items-center justify-center px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full space-y-8 text-center">
       <div class="mx-auto flex items-center justify-center h-24 w-24 rounded-full bg-green-100">
         <svg class="h-12 w-12 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
       </div>
       
       <h2 class="mt-6 text-3xl font-extrabold text-gray-900">Appointment Confirmed!</h2>
       
       <p class="mt-2 text-sm text-gray-600">
         Your appointment has been successfully scheduled.
       </p>

       <div class="bg-gray-50 rounded-lg p-4 mt-8">
          <p class="text-xs text-gray-500 uppercase tracking-wide">Confirmation Number</p>
          <p class="text-2xl font-mono font-bold text-[#005DAA] mt-1">{{ store.confirmation_number }}</p>
       </div>

       <div class="mt-8 space-y-4">
          <button
            id="success-go-dashboard"
            @click="handleGoDashboard"
            class="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-[#005DAA] hover:bg-[#004a87] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#005DAA]"
          >
            Go to Dashboard
          </button>
          
          <button
            id="success-go-home"
            @click="handleGoHome"
            class="w-full flex justify-center py-3 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none"
          >
            Return Home
          </button>
       </div>
    </div>
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SCHEDULE_VISIT_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    onMounted(() => {
      // Set confirmation number if not set (Mocking effect logic here for simplicity if not triggered by action transition)
      // Actually effect is on ACT_SCHED_SUCCESS_GO_HOME but we display it before?
      // Wait, FSM says "ACT_SCHED_SUCCESS_GO_HOME" sets "confirmation_number" to "CONF123".
      // But we need to display it BEFORE clicking home.
      // Usually effects happen on transition. If the value is needed for display, it should have been set by the previous action (CONFIRM).
      // Checking FSM... ACT_SCHED_REVIEW_CONFIRM has NO effects.
      // ACT_SCHED_SUCCESS_GO_HOME has effect setting it.
      // This implies the confirmation number is generated/shown only when leaving? Or maybe the FSM definition assumes it's generated on entry?
      // Since I need to display it, I will set it on mount as a mock simulation, or just use the store value if it was set (it's null initially).
      // I'll set it here to ensure it's visible.
      store.confirmation_number = "CONF123"
    })

    const handleGoDashboard = async () => {
      // ACT_SCHED_SUCCESS_GO_DASHBOARD
      store.setCurrentPageId('DASHBOARD')
      await router.push({ name: 'DASHBOARD' })
    }

    const handleGoHome = async () => {
      // ACT_SCHED_SUCCESS_GO_HOME
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      handleGoDashboard,
      handleGoHome
    }
  }
}
</script>