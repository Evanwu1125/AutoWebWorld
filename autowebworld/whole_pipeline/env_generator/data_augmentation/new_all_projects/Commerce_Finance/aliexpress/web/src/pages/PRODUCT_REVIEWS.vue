<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Header -->
    <header class="bg-white shadow-sm px-4 py-3 flex items-center space-x-3 sticky top-0 z-20">
      <button 
        id="reviews-back-product" 
        class="p-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackDetail"
      >
        <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Ratings & Reviews</h1>
    </header>

    <!-- Summary -->
    <div class="bg-white mt-2 p-4 shadow-sm">
      <div class="flex items-center justify-between mb-4">
        <div>
           <div class="text-4xl font-black text-gray-900">4.8<span class="text-lg text-gray-400 font-medium">/5</span></div>
           <div class="text-yellow-400 text-sm mt-1">★★★★★</div>
           <div class="text-xs text-gray-500 mt-1">2,341 Ratings</div>
        </div>
        <div class="flex-1 ml-8 space-y-1">
           <div class="flex items-center text-xs">
              <span class="w-8 text-gray-500">5 ★</span>
              <div class="flex-1 h-2 bg-gray-100 rounded-full mx-2"><div class="bg-yellow-400 h-2 rounded-full" style="width: 85%"></div></div>
              <span class="text-gray-400">85%</span>
           </div>
           <div class="flex items-center text-xs">
              <span class="w-8 text-gray-500">4 ★</span>
              <div class="flex-1 h-2 bg-gray-100 rounded-full mx-2"><div class="bg-yellow-400 h-2 rounded-full" style="width: 10%"></div></div>
              <span class="text-gray-400">10%</span>
           </div>
           <div class="flex items-center text-xs">
              <span class="w-8 text-gray-500">3 ★</span>
              <div class="flex-1 h-2 bg-gray-100 rounded-full mx-2"><div class="bg-yellow-400 h-2 rounded-full" style="width: 3%"></div></div>
              <span class="text-gray-400">3%</span>
           </div>
           <div class="flex items-center text-xs">
              <span class="w-8 text-gray-500">2 ★</span>
              <div class="flex-1 h-2 bg-gray-100 rounded-full mx-2"><div class="bg-yellow-400 h-2 rounded-full" style="width: 1%"></div></div>
              <span class="text-gray-400">1%</span>
           </div>
           <div class="flex items-center text-xs">
              <span class="w-8 text-gray-500">1 ★</span>
              <div class="flex-1 h-2 bg-gray-100 rounded-full mx-2"><div class="bg-yellow-400 h-2 rounded-full" style="width: 1%"></div></div>
              <span class="text-gray-400">1%</span>
           </div>
        </div>
      </div>
    </div>

    <!-- Reviews List -->
    <div id="reviews-list-container" class="mt-2 bg-white shadow-sm">
      <div 
        v-for="i in 5" 
        :key="i" 
        class="p-4 border-b border-gray-50"
        :class="`data-id-review-${i}`"
        @click="handleReviewClick(i)"
      >
         <div class="flex justify-between items-start mb-2">
            <div class="flex items-center space-x-2">
               <div class="w-8 h-8 bg-gray-200 rounded-full"></div>
               <span class="text-sm font-medium text-gray-900">User {{ i }}</span>
            </div>
            <span class="text-xs text-gray-400">2 days ago</span>
         </div>
         <div class="text-yellow-400 text-xs mb-2">★★★★★</div>
         <div class="text-sm text-gray-600 mb-2">
            Great product! Fast shipping and good quality. Exactly as described. Will buy again.
         </div>
         <div class="flex space-x-2 mt-2 overflow-x-auto">
            <div class="w-16 h-16 bg-gray-100 rounded-md flex-shrink-0"></div>
            <div class="w-16 h-16 bg-gray-100 rounded-md flex-shrink-0"></div>
         </div>
         <div class="text-xs text-gray-400 mt-2">Color: Black, Size: L</div>
      </div>
    </div>

  </div>
</template>

<script>
import { watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PRODUCT_REVIEWS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const handleBackDetail = async () => {
       signatureStore.currentPageId = 'PRODUCT_DETAIL'
       await router.push({ name: 'PRODUCT_DETAIL' })
    }

    const handleReviewClick = (id) => {
       // For scroll into view action mostly
       signatureStore.PRODUCT_REVIEWS_viewport_anchor_id = `review-${id}`
    }
    
    watch(() => signatureStore.PRODUCT_REVIEWS_viewport_anchor_id, async (newId) => {
      if (newId) {
        await nextTick()
        const element = document.querySelector(`.data-id-${newId}`)
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' })
        }
      }
    })

    return {
       handleBackDetail,
       handleReviewClick
    }
  }
}
</script>