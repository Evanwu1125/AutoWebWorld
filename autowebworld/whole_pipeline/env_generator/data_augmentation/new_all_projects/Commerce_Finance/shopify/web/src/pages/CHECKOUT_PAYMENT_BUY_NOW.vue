<template>
  <div class="min-h-screen bg-white font-sans flex flex-col items-center justify-center p-4">
    <div class="w-full max-w-md space-y-8">
        <div class="text-center">
            <h1 class="text-2xl font-bold text-gray-900">Buy Now Payment</h1>
        </div>

        <div class="space-y-6">
            <div>
                <label class="block text-sm font-medium text-gray-700 mb-1">Card Number</label>
                <input 
                    id="payment-card-number-buy-now"
                    type="text" 
                    v-model="cardNumber" 
                    placeholder="Card Number"
                    class="w-full border-gray-300 rounded-lg shadow-sm focus:border-[#008060] focus:ring focus:ring-[#008060] py-3 px-4"
                />
            </div>
            
            <div class="flex space-x-4 pt-4">
                 <button 
                    id="payment-back-buy-now" 
                    @click="goBackInfo"
                    class="flex-1 bg-white border border-gray-300 text-gray-700 font-bold py-3 px-4 rounded-lg hover:bg-gray-50"
                >
                    Back
                </button>
                <button 
                    id="payment-pay-now-buy-now" 
                    @click="payNow"
                    class="flex-1 bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-3 px-4 rounded-lg shadow-md"
                >
                    Pay {{ signatureStore.selected_quantity }} Items
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
  name: 'CHECKOUT_PAYMENT_BUY_NOW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const cardNumber = computed({
        get: () => signatureStore.card_number,
        set: (val) => signatureStore.card_number = val
    })
    
    // FSM defines card_name update too, though simplified UI
    // We assume card_name updated via some implicit mechanism or just default for Buy Now 
    // Actually FSM has ACT_CHECKOUT_BUY_NOW_TYPE_CARD with card_name param, but let's stick to strict UI elements from FSM
    // Wait, FSM has ACT_CHECKOUT_BUY_NOW_TYPE_CARD which types BOTH number and name? No, usually split or single action with multiple params.
    // FSM: ACT_CHECKOUT_BUY_NOW_TYPE_CARD -> types card_number and card_name. UI procedure might imply typing into two fields or just one representing 'card details'.
    // Let's check FSM... 
    // ACT_CHECKOUT_BUY_NOW_TYPE_CARD uses selector #payment-card-number-buy-now. 
    // It seems to only have one selector for typing both? Or maybe just card number is enough for this simplified flow.
    // Preconditions check both card_number and card_name length > 0.
    // I should add a hidden or auto-filled name field if the FSM doesn't provide a selector for it, OR check if I missed a selector.
    // FSM: ACT_CHECKOUT_BUY_NOW_TYPE_CARD param: card_name="Buyer BuyNow". Effect sets $.card_name.
    // BUT gui_procedure ONLY interacts with #payment-card-number-buy-now. 
    // This implies a simulation where typing the number also 'fills' the name, or it's a single input handling both (unlikely).
    // Safest bet for FSM compliance: The action sets both. The UI input triggers the action.
    
    const goBackInfo = async () => {
        signatureStore.currentPageId = 'CHECKOUT_INFORMATION_BUY_NOW'
        await router.push({ name: 'CHECKOUT_INFORMATION_BUY_NOW' })
    }

    const payNow = async () => {
        if (cardNumber.value) {
            // Implicitly set name to satisfy precondition if not set
            if (!signatureStore.card_name) signatureStore.card_name = "Buyer BuyNow"
            
            signatureStore.order_id = `BN-${Math.floor(Math.random() * 10000)}`
            signatureStore.currentPageId = 'CHECKOUT_SUCCESS_BUY_NOW'
            await router.push({ name: 'CHECKOUT_SUCCESS_BUY_NOW' })
        } else {
            alert('Enter card details')
        }
    }

    return {
        signatureStore,
        cardNumber,
        goBackInfo,
        payNow
    }
  }
}
</script>