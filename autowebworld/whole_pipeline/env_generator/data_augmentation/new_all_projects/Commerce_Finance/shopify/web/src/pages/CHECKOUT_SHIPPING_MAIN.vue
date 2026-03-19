<template>
  <div class="min-h-screen bg-gray-50 flex flex-col md:flex-row text-gray-900 font-sans">
    <!-- Summary Sidebar (Right on Desktop) - Duplicate logic for consistency -->
    <div class="md:w-1/2 bg-gray-50 p-8 md:p-12 order-1 md:order-2 border-l border-gray-200">
         <div class="max-w-lg mx-auto sticky top-12">
             <div class="flex items-center gap-4 mb-6">
                 <div class="text-2xl font-bold text-gray-900">Order Summary</div>
                 <div class="text-gray-500 text-sm">({{ cartItems.length }} items)</div>
             </div>
             <div class="space-y-4 mb-8">
                 <div v-for="item in cartItems" :key="item.id" class="flex items-center gap-4">
                     <div class="relative w-16 h-16 bg-white border border-gray-200 rounded-lg overflow-hidden flex-shrink-0">
                         <img :src="item.image" :alt="item.title" class="w-full h-full object-cover" />
                         <span class="absolute -top-2 -right-2 bg-gray-500 text-white text-xs font-bold w-5 h-5 rounded-full flex items-center justify-center">{{ item.quantity }}</span>
                     </div>
                     <div class="flex-1">
                         <div class="font-medium text-gray-900">{{ item.title }}</div>
                         <div class="text-sm text-gray-500">{{ item.variant_title }}</div>
                     </div>
                     <div class="font-medium text-gray-900">${{ (item.price * item.quantity).toFixed(2) }}</div>
                 </div>
             </div>
             <div class="border-t border-gray-200 pt-6 space-y-2">
                 <div class="flex justify-between text-gray-600">
                     <span>Subtotal</span>
                     <span>${{ subtotal.toFixed(2) }}</span>
                 </div>
                 <div class="flex justify-between text-gray-600">
                     <span>Shipping</span>
                     <span class="text-xs text-gray-900 font-medium">Free</span>
                 </div>
             </div>
             <div class="border-t border-gray-200 mt-6 pt-6 flex justify-between items-center">
                 <span class="text-xl font-bold text-gray-900">Total</span>
                 <span class="text-2xl font-bold text-gray-900">${{ subtotal.toFixed(2) }}</span>
             </div>
         </div>
    </div>

    <!-- Main Content -->
    <div class="md:w-1/2 bg-white p-8 md:p-12 order-2 md:order-1">
        <div class="max-w-lg mx-auto">
            <h1 class="text-2xl font-bold text-[#008060] mb-8 tracking-tight">STOREFRONT</h1>
            
            <nav class="flex items-center text-xs md:text-sm text-gray-500 mb-8 font-medium">
                <span class="text-[#008060]">Information</span>
                <span class="mx-2">›</span>
                <span class="text-[#008060]">Shipping</span>
                <span class="mx-2">›</span>
                <span>Payment</span>
            </nav>

            <div class="space-y-8">
                <!-- Review Info Box -->
                <div class="border border-gray-200 rounded-lg p-4 text-sm text-gray-600 space-y-3">
                    <div class="flex justify-between pb-3 border-b border-gray-100">
                        <span class="text-gray-500 w-16">Contact</span>
                        <span class="font-medium text-gray-900 flex-1 ml-4">{{ signatureStore.email }}</span>
                        <span class="text-[#008060] cursor-pointer hover:underline text-xs" @click="goBackInfo">Change</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500 w-16">Ship to</span>
                        <span class="font-medium text-gray-900 flex-1 ml-4">{{ signatureStore.shipping_address1 }}, {{ signatureStore.shipping_city }} {{ signatureStore.shipping_postcode }}</span>
                        <span class="text-[#008060] cursor-pointer hover:underline text-xs" @click="goBackInfo">Change</span>
                    </div>
                </div>

                <!-- Shipping Method -->
                <section>
                    <h2 class="text-lg font-bold text-gray-900 mb-4">Shipping method</h2>
                    <div class="border border-gray-200 rounded-lg overflow-hidden">
                        <label class="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-50 border-b border-gray-200 last:border-0" id="shipping-method-standard">
                             <div class="flex items-center">
                                 <input 
                                    type="radio" 
                                    value="standard" 
                                    v-model="shippingMethod" 
                                    class="h-4 w-4 text-[#008060] focus:ring-[#008060] border-gray-300"
                                 />
                                 <div class="ml-3">
                                     <div class="font-medium text-gray-900">Standard</div>
                                     <div class="text-xs text-gray-500">5 to 8 business days</div>
                                 </div>
                             </div>
                             <span class="font-medium text-gray-900">Free</span>
                        </label>
                         <label class="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-50" id="shipping-method-express">
                             <div class="flex items-center">
                                 <input 
                                    type="radio" 
                                    value="express" 
                                    v-model="shippingMethod" 
                                    class="h-4 w-4 text-[#008060] focus:ring-[#008060] border-gray-300"
                                 />
                                 <div class="ml-3">
                                     <div class="font-medium text-gray-900">Express</div>
                                     <div class="text-xs text-gray-500">2 to 3 business days</div>
                                 </div>
                             </div>
                             <span class="font-medium text-gray-900">$15.00</span>
                        </label>
                    </div>
                </section>

                <!-- Actions -->
                <div class="flex items-center justify-between pt-6 border-t border-gray-100">
                    <span 
                        id="shipping-back-to-information" 
                        @click="goBackInfo" 
                        class="text-[#008060] cursor-pointer hover:underline text-sm font-medium flex items-center"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                        </svg>
                        Return to information
                    </span>
                    <button 
                        id="shipping-continue-to-payment" 
                        @click="continueToPayment"
                        class="bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-4 px-8 rounded-lg shadow-md transition-all"
                    >
                        Continue to payment
                    </button>
                </div>
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
  name: 'CHECKOUT_SHIPPING_MAIN',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const cartItems = computed(() => signatureStore.cart_items)
    const subtotal = computed(() => signatureStore.cart_subtotal)
    
    const shippingMethod = computed({
        get: () => signatureStore.shipping_method,
        set: (val) => signatureStore.shipping_method = val
    })

    const goBackInfo = async () => {
        signatureStore.currentPageId = 'CHECKOUT_INFORMATION_MAIN'
        await router.push({ name: 'CHECKOUT_INFORMATION_MAIN' })
    }

    const continueToPayment = async () => {
        if (shippingMethod.value) {
            signatureStore.currentPageId = 'CHECKOUT_PAYMENT_MAIN'
            await router.push({ name: 'CHECKOUT_PAYMENT_MAIN' })
        } else {
             alert('Please select a shipping method.')
        }
    }

    return {
        signatureStore,
        cartItems,
        subtotal,
        shippingMethod,
        goBackInfo,
        continueToPayment
    }
  }
}
</script>