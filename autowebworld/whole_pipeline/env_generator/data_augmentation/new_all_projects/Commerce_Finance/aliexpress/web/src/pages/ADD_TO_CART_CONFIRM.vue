<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-md p-6 text-center">
      <div class="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-green-100 mb-6">
        <svg class="h-8 w-8 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-900 mb-2">Added to Cart!</h2>
      <p class="text-gray-600 mb-8">Item has been successfully added to your shopping cart.</p>
      
      <div class="space-y-3">
        <button 
          id="go-to-cart-button" 
          class="w-full bg-red-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-red-700 transition-colors shadow-md"
          @click="handleGoCart"
        >
          View Shopping Cart
        </button>
        
        <button 
          id="continue-shopping-button" 
          class="w-full bg-white text-gray-700 font-bold py-3 px-4 rounded-lg border border-gray-300 hover:bg-gray-50 transition-colors"
          @click="handleContinueShopping"
        >
          Continue Shopping
        </button>
        
        <button 
          id="add-to-cart-back-product" 
          class="text-sm text-gray-500 hover:text-gray-700 underline"
          @click="handleBackDetail"
        >
          Back to Product Details
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ADD_TO_CART_CONFIRM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const handleGoCart = async () => {
       signatureStore.currentPageId = 'CART_PAGE'
       await router.push({ name: 'CART_PAGE' })
    }

    const handleContinueShopping = async () => {
       signatureStore.currentPageId = 'PRODUCT_LIST'
       await router.push({ name: 'PRODUCT_LIST' })
    }

    const handleBackDetail = async () => {
       signatureStore.currentPageId = 'PRODUCT_DETAIL'
       await router.push({ name: 'PRODUCT_DETAIL' })
    }

    return {
       handleGoCart,
       handleContinueShopping,
       handleBackDetail
    }
  }
}
</script>