<template>
  <div class="min-h-screen bg-white flex flex-col">
    <nav class="border-b border-gray-200 p-4">
       <button id="membership-back-settings" @click="handleBack" class="text-gray-500 hover:text-black font-sans text-sm flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back to Settings
       </button>
    </nav>

    <div class="flex-1 flex flex-col items-center justify-center p-8 max-w-4xl mx-auto w-full">
       <h1 class="text-5xl md:text-6xl font-serif font-bold text-center mb-6">Fuel great thinking.</h1>
       <p class="text-xl font-sans text-gray-500 text-center mb-16 max-w-xl">Become a Medium member to enjoy unlimited access and directly support the writers you read.</p>
       
       <div class="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-3xl">
          <!-- Monthly -->
          <div 
             id="membership-plan-monthly" 
             @click="selectPlan('monthly')"
             :class="{
                'border-2 rounded-xl p-8 cursor-pointer transition-all relative hover:shadow-xl': true,
                'border-black bg-gray-50': selectedPlan === 'monthly',
                'border-gray-200 hover:border-gray-400': selectedPlan !== 'monthly'
             }"
          >
             <div class="text-2xl font-bold font-serif mb-2">Monthly</div>
             <div class="text-4xl font-sans font-bold mb-6">$5<span class="text-lg font-normal text-gray-500">/month</span></div>
             <ul class="space-y-3 text-sm font-sans text-gray-600 mb-8">
                <li class="flex items-center gap-2">✓ Unlimited access to every story</li>
                <li class="flex items-center gap-2">✓ Support writers directly</li>
                <li class="flex items-center gap-2">✓ Cancel anytime</li>
             </ul>
             <div v-if="selectedPlan === 'monthly'" class="absolute top-4 right-4 text-green-600">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                </svg>
             </div>
          </div>

          <!-- Yearly -->
          <div 
             id="membership-plan-yearly" 
             @click="selectPlan('yearly')"
             :class="{
                'border-2 rounded-xl p-8 cursor-pointer transition-all relative hover:shadow-xl': true,
                'border-black bg-gray-50': selectedPlan === 'yearly',
                'border-gray-200 hover:border-gray-400': selectedPlan !== 'yearly'
             }"
          >
             <div class="absolute -top-3 right-8 bg-green-600 text-white text-xs font-bold px-3 py-1 rounded-full font-sans">BEST VALUE</div>
             <div class="text-2xl font-bold font-serif mb-2">Yearly</div>
             <div class="text-4xl font-sans font-bold mb-6">$50<span class="text-lg font-normal text-gray-500">/year</span></div>
             <ul class="space-y-3 text-sm font-sans text-gray-600 mb-8">
                <li class="flex items-center gap-2">✓ Save $10 a year</li>
                <li class="flex items-center gap-2">✓ Unlimited access to every story</li>
                <li class="flex items-center gap-2">✓ Support writers directly</li>
             </ul>
             <div v-if="selectedPlan === 'yearly'" class="absolute top-4 right-4 text-green-600">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" viewBox="0 0 20 20" fill="currentColor">
                  <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                </svg>
             </div>
          </div>
       </div>

       <div class="mt-12 w-full max-w-3xl flex justify-center">
          <button 
             id="membership-continue" 
             @click="handleContinue" 
             :disabled="!selectedPlan"
             :class="{
                'px-12 py-4 rounded-full text-lg font-medium font-sans transition-all transform': true,
                'bg-black text-white hover:scale-105 shadow-lg': selectedPlan,
                'bg-gray-200 text-gray-400 cursor-not-allowed': !selectedPlan
             }"
          >
             Continue to Payment
          </button>
       </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'MEMBERSHIP_LANDING',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    
    const selectedPlan = ref(null)

    const selectPlan = (plan) => {
       selectedPlan.value = plan
       signatureStore.membership_plan_selected = plan
    }

    const handleContinue = async () => {
       if (selectedPlan.value) {
          signatureStore.setCurrentPageId('PAYMENT_DETAILS')
          await router.push({ name: 'PAYMENT_DETAILS' })
       }
    }

    const handleBack = async () => {
       signatureStore.setCurrentPageId('SETTINGS_PREFERENCES')
       await router.push({ name: 'SETTINGS_PREFERENCES' })
    }

    return {
       selectedPlan,
       selectPlan,
       handleContinue,
       handleBack
    }
  }
}
</script>