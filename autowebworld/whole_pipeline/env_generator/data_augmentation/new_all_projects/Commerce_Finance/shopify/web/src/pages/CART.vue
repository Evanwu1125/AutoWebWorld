<template>
  <div class="min-h-screen bg-gray-50 text-gray-900 font-sans">
    <nav class="bg-white border-b border-gray-200">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <span class="text-xl font-bold text-[#008060]">Your Cart</span>
            <span id="back-home-from-cart" @click="goHome" class="text-gray-500 hover:text-[#008060] cursor-pointer text-sm font-medium">
                Continue Shopping
            </span>
        </div>
    </nav>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div v-if="cartItems.length > 0" class="flex flex-col lg:flex-row gap-12">
            <!-- Cart Items -->
            <div class="flex-1 space-y-6">
                <div v-for="(item, index) in cartItems" :key="item.id" class="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex gap-6 items-start relative">
                     <div class="w-24 h-24 bg-gray-100 rounded-lg overflow-hidden flex-shrink-0">
                         <img :src="item.image" :alt="item.title" class="w-full h-full object-cover" />
                     </div>
                     <div class="flex-1">
                         <h3 class="font-bold text-lg text-gray-900">{{ item.title }}</h3>
                         <div class="text-sm text-gray-500 mb-2">{{ item.variant_title }}</div>
                         <div class="font-bold text-gray-900">${{ item.price.toFixed(2) }}</div>
                         
                         <div class="mt-4 flex items-center space-x-4">
                             <div class="flex items-center border border-gray-300 rounded-lg">
                                 <input 
                                    :id="index === 0 ? 'cart-line-1-qty' : `cart-line-${index+1}-qty`"
                                    type="number" 
                                    v-model.number="item.quantity" 
                                    @input="updateQuantity"
                                    min="1"
                                    class="w-16 py-1 px-2 text-center border-none focus:ring-0 rounded-lg"
                                 />
                             </div>
                             <button 
                                :id="index === 0 ? 'cart-line-1-remove' : `cart-line-${index+1}-remove`"
                                @click="removeItem(item.id)"
                                class="text-red-500 hover:text-red-700 text-sm font-medium underline"
                             >
                                Remove
                             </button>
                         </div>
                     </div>
                     <div class="font-bold text-lg text-gray-900">
                         ${{ (item.price * item.quantity).toFixed(2) }}
                     </div>
                </div>
            </div>

            <!-- Summary -->
            <div class="w-full lg:w-96 flex-shrink-0">
                <div class="bg-white p-8 rounded-xl shadow-sm border border-gray-100 sticky top-24">
                    <h2 class="text-xl font-bold text-gray-900 mb-6">Order Summary</h2>
                    <div class="flex justify-between mb-4 pb-4 border-b border-gray-100">
                        <span class="text-gray-600">Subtotal</span>
                        <span class="font-bold text-gray-900">${{ subtotal.toFixed(2) }}</span>
                    </div>
                    <div class="text-sm text-gray-500 mb-8">Shipping and taxes calculated at checkout.</div>
                    
                    <div class="space-y-4">
                        <button 
                            id="checkout-button" 
                            @click="checkout"
                            class="w-full bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-4 px-6 rounded-lg shadow-md transition-colors"
                        >
                            Check Out
                        </button>
                        
                        <div class="relative py-2">
                            <div class="absolute inset-0 flex items-center">
                                <div class="w-full border-t border-gray-300"></div>
                            </div>
                            <div class="relative flex justify-center">
                                <span class="bg-white px-2 text-sm text-gray-500 uppercase">Or express checkout</span>
                            </div>
                        </div>

                        <button 
                            id="express-checkout-button" 
                            @click="expressCheckout"
                            class="w-full bg-[#5A31F4] hover:bg-[#4825C9] text-white font-bold py-4 px-6 rounded-lg shadow-md transition-colors flex items-center justify-center gap-2"
                        >
                            <span>⚡ Shop Pay</span>
                        </button>
                    </div>
                    
                    <button 
                        id="continue-shopping" 
                        @click="continueShopping"
                        class="w-full mt-6 text-[#008060] font-medium hover:underline"
                    >
                        Continue Shopping
                    </button>
                </div>
            </div>
        </div>
        
        <div v-else class="text-center py-24 bg-white rounded-xl shadow-sm border border-gray-100">
             <div class="text-6xl mb-6">🛒</div>
             <h2 class="text-2xl font-bold text-gray-900 mb-4">Your cart is empty</h2>
             <p class="text-gray-500 mb-8">Looks like you haven't added anything yet.</p>
             <button 
                id="back-home-from-cart" 
                @click="goHome"
                class="bg-[#008060] hover:bg-[#004C3F] text-white font-bold py-3 px-8 rounded-lg transition-colors"
             >
                Start Shopping
             </button>
        </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'CART',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const cartItems = computed(() => signatureStore.cart_items)
    const subtotal = computed(() => signatureStore.cart_subtotal)

    const updateQuantity = () => {
        // Recalculate subtotal
        signatureStore.cart_subtotal = signatureStore.cart_items.reduce((sum, item) => sum + (item.price * item.quantity), 0)
    }

    const removeItem = (itemId) => {
        const index = signatureStore.cart_items.findIndex(i => i.id === itemId)
        if (index !== -1) {
            signatureStore.cart_items.splice(index, 1)
            updateQuantity()
        }
    }

    const checkout = async () => {
        if (cartItems.value.length === 0) return
        signatureStore.currentPageId = 'CHECKOUT_INFORMATION_MAIN'
        await router.push({ name: 'CHECKOUT_INFORMATION_MAIN' })
    }

    const expressCheckout = async () => {
        if (cartItems.value.length === 0) return
        signatureStore.currentPageId = 'CHECKOUT_EXPRESS'
        await router.push({ name: 'CHECKOUT_EXPRESS' })
    }

    const continueShopping = async () => {
        signatureStore.currentPageId = 'SHOP_ALL_COLLECTIONS'
        await router.push({ name: 'SHOP_ALL_COLLECTIONS' })
    }

    const goHome = async () => {
        signatureStore.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        cartItems,
        subtotal,
        updateQuantity,
        removeItem,
        checkout,
        expressCheckout,
        continueShopping,
        goHome
    }
  }
}
</script>