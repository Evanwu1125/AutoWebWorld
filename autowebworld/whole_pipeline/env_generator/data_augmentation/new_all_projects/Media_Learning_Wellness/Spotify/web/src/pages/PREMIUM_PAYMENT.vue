<template>
  <div class="flex h-screen bg-[#121212] text-white font-sans overflow-hidden">
    <main class="flex-1 overflow-y-auto p-8 md:p-12 max-w-2xl mx-auto w-full flex flex-col justify-center">
      <div id="back-premium-upsell" @click="handleBackUpsell" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8 self-start">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
         <span>Back to Plans</span>
      </div>

      <div class="bg-[#181818] p-8 rounded-xl border border-[#282828] shadow-2xl">
         <h1 class="text-3xl font-bold mb-8 text-center">Checkout</h1>
         
         <div class="space-y-6">
            <!-- Card Number -->
            <div>
               <label class="block text-xs font-bold uppercase text-[#B3B3B3] mb-2">Card Number</label>
               <input 
                 id="card-number-input"
                 v-model="form.cardNumber"
                 @input="handleInputCardNumber"
                 type="text" 
                 placeholder="0000 0000 0000 0000"
                 class="w-full bg-[#282828] border border-transparent focus:border-[#1DB954] rounded p-3 text-white placeholder-[#535353] outline-none font-mono transition-colors"
               />
            </div>

            <div class="grid grid-cols-2 gap-6">
               <!-- Expiry -->
               <div>
                  <label class="block text-xs font-bold uppercase text-[#B3B3B3] mb-2">Expiry Date</label>
                  <input 
                    id="card-expiry-input"
                    v-model="form.cardExpiry"
                    @input="handleInputExpiry"
                    type="text" 
                    placeholder="MM/YY"
                    class="w-full bg-[#282828] border border-transparent focus:border-[#1DB954] rounded p-3 text-white placeholder-[#535353] outline-none font-mono transition-colors"
                  />
               </div>
               <!-- CVC -->
               <div>
                  <label class="block text-xs font-bold uppercase text-[#B3B3B3] mb-2">Security Code</label>
                  <input 
                    id="card-cvc-input"
                    v-model="form.cardCvc"
                    @input="handleInputCvc"
                    type="text" 
                    placeholder="CVC"
                    class="w-full bg-[#282828] border border-transparent focus:border-[#1DB954] rounded p-3 text-white placeholder-[#535353] outline-none font-mono transition-colors"
                  />
               </div>
            </div>

            <!-- Billing Name -->
            <div>
               <label class="block text-xs font-bold uppercase text-[#B3B3B3] mb-2">Name on Card</label>
               <input 
                 id="billing-name-input"
                 v-model="form.billingName"
                 @input="handleInputName"
                 type="text" 
                 placeholder="Name on card"
                 class="w-full bg-[#282828] border border-transparent focus:border-[#1DB954] rounded p-3 text-white placeholder-[#535353] outline-none transition-colors"
               />
            </div>
         </div>

         <div class="mt-8 pt-8 border-t border-[#282828]">
            <div class="flex justify-between items-center mb-6 text-sm">
               <span class="text-[#B3B3B3]">Total today</span>
               <span class="font-bold text-xl">$9.99</span>
            </div>
            
            <button 
               id="premium-pay-submit"
               @click="handleSubmit"
               class="w-full bg-[#1DB954] hover:bg-[#1ed760] text-black font-bold py-4 rounded-full uppercase tracking-widest hover:scale-105 transition-transform shadow-lg"
            >
               Pay Now
            </button>
         </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useRouter } from 'vue-router'

export default {
  name: 'PREMIUM_PAYMENT',
  setup() {
    const store = useSignatureStore()
    const router = useRouter()

    const form = ref({
       cardNumber: '',
       cardExpiry: '',
       cardCvc: '',
       billingName: ''
    })

    const handleBackUpsell = async () => {
       store.setCurrentPageId('PREMIUM_UPSELL')
       await router.push({ name: 'PREMIUM_UPSELL' })
    }

    const handleInputCardNumber = () => store.card_number = form.value.cardNumber
    const handleInputExpiry = () => store.card_expiry = form.value.cardExpiry
    const handleInputCvc = () => store.card_cvc = form.value.cardCvc
    const handleInputName = () => store.billing_name = form.value.billingName

    const handleSubmit = async () => {
       if (store.card_number && store.card_expiry && store.card_cvc && store.billing_name) {
          store.setCurrentPageId('PREMIUM_UPGRADE_SUCCESS')
          await router.push({ name: 'PREMIUM_UPGRADE_SUCCESS' })
       }
    }

    return {
       form,
       handleBackUpsell,
       handleInputCardNumber,
       handleInputExpiry,
       handleInputCvc,
       handleInputName,
       handleSubmit
    }
  }
}
</script>