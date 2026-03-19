<template>
  <div class="flex h-screen bg-[#121212] text-white font-sans overflow-hidden">
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div id="back-account-overview" @click="handleBackAccount" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
         <span>Back to Account</span>
      </div>
    </aside>

    <main class="flex-1 overflow-y-auto p-8 md:p-12 max-w-5xl mx-auto w-full">
      <div class="text-center mb-12">
         <h1 class="text-5xl font-bold mb-4">Pick your Premium</h1>
         <p class="text-xl text-[#B3B3B3]">Listen without limits on your phone, speaker, and other devices.</p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
         <!-- Individual Plan -->
         <div 
           id="premium-plan-individual-card"
           @click="handleSelectPlan('premium_individual')"
           class="bg-[#242424] rounded-xl p-8 border-2 cursor-pointer transition-all hover:scale-105"
           :class="selectedPlan === 'premium_individual' ? 'border-[#1DB954] shadow-[0_0_20px_rgba(29,185,84,0.3)]' : 'border-transparent'"
         >
            <div class="bg-[#ffd2d7] text-black text-xs font-bold inline-block px-2 py-1 rounded mb-4">One-time payment</div>
            <h3 class="text-2xl font-bold mb-2">Individual</h3>
            <p class="mb-6">$9.99/month after offer period</p>
            <hr class="border-[#3E3E3E] mb-6" />
            <ul class="space-y-3 mb-8 text-sm">
               <li class="flex items-start"><svg class="w-5 h-5 mr-2 text-[#1DB954]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg> Ad-free music listening</li>
               <li class="flex items-start"><svg class="w-5 h-5 mr-2 text-[#1DB954]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg> Play anywhere - even offline</li>
               <li class="flex items-start"><svg class="w-5 h-5 mr-2 text-[#1DB954]" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg> On-demand playback</li>
            </ul>
            <button class="w-full bg-[#ffd2d7] text-black font-bold py-3 rounded-full uppercase tracking-widest text-sm hover:bg-white transition-colors">
               Select
            </button>
         </div>

         <!-- Duo Plan (Visual Only) -->
         <div class="bg-[#242424] rounded-xl p-8 border-2 border-transparent opacity-60">
            <div class="bg-[#ffc862] text-black text-xs font-bold inline-block px-2 py-1 rounded mb-4">One-time payment</div>
            <h3 class="text-2xl font-bold mb-2">Duo</h3>
            <p class="mb-6">$12.99/month</p>
            <hr class="border-[#3E3E3E] mb-6" />
            <ul class="space-y-3 mb-8 text-sm">
               <li>2 Premium accounts</li>
               <li>For couples under one roof</li>
            </ul>
         </div>

          <!-- Family Plan (Visual Only) -->
         <div class="bg-[#242424] rounded-xl p-8 border-2 border-transparent opacity-60">
            <div class="bg-[#a5bbd1] text-black text-xs font-bold inline-block px-2 py-1 rounded mb-4">One-time payment</div>
            <h3 class="text-2xl font-bold mb-2">Family</h3>
            <p class="mb-6">$15.99/month</p>
            <hr class="border-[#3E3E3E] mb-6" />
            <ul class="space-y-3 mb-8 text-sm">
               <li>6 Premium accounts</li>
               <li>Block explicit music</li>
            </ul>
         </div>
      </div>

      <!-- Continue Button -->
      <div v-if="selectedPlan" class="fixed bottom-0 left-0 w-full bg-[#181818] border-t border-[#282828] p-4 flex justify-center z-50 animate-slide-up">
         <button 
           id="premium-continue-button"
           @click="handleContinue"
           class="bg-[#1DB954] text-black font-bold py-4 px-12 rounded-full hover:scale-105 transition-transform uppercase tracking-widest text-sm shadow-lg"
         >
            Continue to Payment
         </button>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useRouter } from 'vue-router'

export default {
  name: 'PREMIUM_UPSELL',
  setup() {
    const store = useSignatureStore()
    const router = useRouter()

    const selectedPlan = computed(() => store.selected_premium_plan)

    const handleBackAccount = async () => {
       store.setCurrentPageId('ACCOUNT_OVERVIEW')
       await router.push({ name: 'ACCOUNT_OVERVIEW' })
    }

    const handleSelectPlan = (plan) => {
       store.selected_premium_plan = plan
    }

    const handleContinue = async () => {
       if (selectedPlan.value) {
          store.setCurrentPageId('PREMIUM_PAYMENT')
          await router.push({ name: 'PREMIUM_PAYMENT' })
       }
    }

    return {
       selectedPlan,
       handleBackAccount,
       handleSelectPlan,
       handleContinue
    }
  }
}
</script>

<style scoped>
@keyframes slide-up {
  from { transform: translateY(100%); }
  to { transform: translateY(0); }
}
.animate-slide-up {
  animation: slide-up 0.3s ease-out forwards;
}
</style>