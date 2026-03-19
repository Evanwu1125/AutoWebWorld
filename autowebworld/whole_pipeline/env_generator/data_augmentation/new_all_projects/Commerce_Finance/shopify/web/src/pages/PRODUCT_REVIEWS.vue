<template>
  <div class="min-h-screen bg-gray-50 text-gray-900 font-sans">
     <nav class="bg-white border-b border-gray-200 sticky top-0 z-30">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <span id="back-to-product" @click="goBackProduct" class="text-gray-500 hover:text-[#008060] cursor-pointer flex items-center font-medium">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
                </svg>
                Back to Product
            </span>
             <span id="back-to-collections-from-reviews" @click="goBackCollections" class="text-gray-500 hover:text-[#008060] cursor-pointer text-sm">
                Or continue shopping
            </span>
        </div>
    </nav>

    <main class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <h1 class="text-3xl font-bold text-gray-900 mb-8">Customer Reviews</h1>
        
        <div id="reviews-list" class="space-y-6" @click="scrollReviews">
            <!-- Mock Reviews -->
            <div v-for="i in 5" :key="i" class="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <div class="flex items-center justify-between mb-4">
                    <div class="flex items-center space-x-3">
                         <div class="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center font-bold text-gray-600">
                             {{ ['JD', 'AS', 'MR', 'KL', 'BP'][i-1] }}
                         </div>
                         <div>
                             <div class="font-bold text-gray-900">{{ ['John Doe', 'Alice Smith', 'Mike Ross', 'Kate Lane', 'Bob Pete'][i-1] }}</div>
                             <div class="text-xs text-gray-500">Verified Buyer</div>
                         </div>
                    </div>
                    <div class="text-yellow-400 text-lg">★★★★★</div>
                </div>
                <h3 class="font-bold text-gray-900 mb-2">Great Product!</h3>
                <p class="text-gray-600">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
                <div class="mt-4 text-xs text-gray-400">Posted 2 days ago</div>
            </div>
            
             <!-- More filler content to enable scrolling -->
             <div v-for="i in 5" :key="i+5" class="bg-white p-6 rounded-xl shadow-sm border border-gray-100 opacity-75">
                <div class="flex items-center justify-between mb-4">
                     <div class="flex items-center space-x-3">
                         <div class="w-10 h-10 rounded-full bg-gray-200 flex items-center justify-center font-bold text-gray-600">User</div>
                         <div>
                             <div class="font-bold text-gray-900">Anonymous</div>
                         </div>
                    </div>
                    <div class="text-yellow-400 text-lg">★★★★☆</div>
                </div>
                <p class="text-gray-600">Good quality but shipping took a bit longer than expected.</p>
            </div>
        </div>
    </main>
  </div>
</template>

<script>
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PRODUCT_REVIEWS',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()

    const goBackProduct = async () => {
        signatureStore.currentPageId = 'PRODUCT_DETAIL'
        await router.push({ name: 'PRODUCT_DETAIL', params: { id: route.params.id } })
    }

    const goBackCollections = async () => {
        signatureStore.currentPageId = 'SHOP_ALL_COLLECTIONS'
        await router.push({ name: 'SHOP_ALL_COLLECTIONS' })
    }

    const scrollReviews = () => {
        // Visual feedback or logic if needed, primarily matches FSM action
        console.log('Scrolling reviews...')
    }

    return {
        goBackProduct,
        goBackCollections,
        scrollReviews
    }
  }
}
</script>