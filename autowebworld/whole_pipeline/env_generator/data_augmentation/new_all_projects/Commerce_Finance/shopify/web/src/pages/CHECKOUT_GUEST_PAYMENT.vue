<template>
  <div class="min-h-screen bg-white font-sans flex flex-col items-center justify-center p-4">
    <div class="w-full max-w-md space-y-8">
        <div class="text-center">
            <h1 class="text-2xl font-bold text-gray-900">Guest Payment</h1>
        </div>

        <div class="space-y-6">
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">Card Number</label>
                <input 
                    id="guest-card-number"
                    type="text" 
                    v-model="cardNumber" 
                    placeholder="Card Number"
                    class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] py-3 px-4"
                />
            </div>

            <div class="flex items-center justify-between pt-4">
                <span 
                    id="guest-back-to-info" 
                    @click="goBackInfo"
                    class="text-[#008060] cursor-pointer hover:underline text-sm font-medium"
                >
                    Back to information
                </span>
                <button 
                    id="guest-pay-now" 
                    @click="payNow"
                    class="bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-3 px-6 rounded-lg shadow-md"
                >
                    Pay Now
                </button>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CHECKOUT_GUEST_PAYMENT',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const cardNumber = computed({
        get: () => signatureStore.card_number,
        set: (val) => signatureStore.card_number = val
    })

    const goBackInfo = async () => {
        signatureStore.currentPageId = 'CHECKOUT_GUEST_INFORMATION'
        await router.push({ name: 'CHECKOUT_GUEST_INFORMATION' })
    }

    const payNow = async () => {
        if (cardNumber.value) {
            // Implicit name set
            signatureStore.card_name = "Guest User"
            signatureStore.order_id = `GST-${Math.floor(Math.random() * 10000)}`
            
            signatureStore.currentPageId = 'CHECKOUT_SUCCESS_GUEST'
            await router.push({ name: 'CHECKOUT_SUCCESS_GUEST' })
        } else {
            alert('Enter card number')
        }
    }

    return {
        cardNumber,
        goBackInfo,
        payNow
    }
  }
}
</script>