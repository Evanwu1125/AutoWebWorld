<template>
  <div class="departments-page min-h-screen flex flex-col bg-gray-50">
    <!-- Simple Header for Departments Page -->
    <header class="bg-white shadow-sm p-4 sticky top-0 z-20">
      <div class="max-w-7xl mx-auto flex items-center justify-between">
        <div id="logo-home" @click="handleGoToHome" class="font-bold text-xl text-[#0071DC] cursor-pointer flex items-center gap-2">
           <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" /></svg>
           Walmart Departments
        </div>
        
        <!-- Dept Menu Toggle -->
        <div id="dept-menu-toggle" class="relative group cursor-pointer px-4 py-2 hover:bg-gray-100 rounded-lg">
          <div class="flex items-center gap-2 font-medium">
            Browse
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
          </div>
          
          <!-- Dropdown Menu -->
          <div class="absolute right-0 top-full mt-2 w-56 bg-white rounded-lg shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none group-hover:pointer-events-auto z-50 p-2">
             <div 
               id="dept-menu-grocery"
               @click="handleGoToGrocery" 
               class="block px-4 py-2 hover:bg-gray-100 rounded text-left cursor-pointer font-medium text-gray-700"
             >
               Groceries
             </div>
             <div class="block px-4 py-2 hover:bg-gray-100 rounded text-left cursor-pointer font-medium text-gray-700">
               Electronics
             </div>
          </div>
        </div>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto w-full p-4 md:p-8">
      <h1 class="text-3xl font-bold mb-8">Shop by Department</h1>
      
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        
        <!-- Electronics Tile -->
        <div 
          id="dept-electronics-tile" 
          @click="handleGoToElectronics"
          class="bg-white rounded-xl shadow-md hover:shadow-xl transition-all cursor-pointer overflow-hidden group border border-gray-100"
        >
          <div class="h-48 overflow-hidden bg-gray-100">
            <img src="/images/photo1766052291.jpg" alt="Electronics" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
          </div>
          <div class="p-6">
            <h2 class="text-xl font-bold mb-2">Electronics</h2>
            <p class="text-gray-600 mb-4">TVs, Laptops, Phones & More</p>
            <span class="text-[#0071DC] font-medium group-hover:underline">Shop Now &rarr;</span>
          </div>
        </div>

        <!-- Grocery Tile (Decorative in this view, as FSM specifies menu navigation for Grocery, but good for visuals) -->
        <div 
          class="bg-white rounded-xl shadow-md hover:shadow-xl transition-all cursor-pointer overflow-hidden group border border-gray-100 opacity-75"
        >
          <div class="h-48 overflow-hidden bg-gray-100">
            <img src="/images/photo1766052291.jpg" alt="Grocery" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
          </div>
          <div class="p-6">
            <h2 class="text-xl font-bold mb-2">Grocery</h2>
            <p class="text-gray-600 mb-4">Fresh Produce & Pantry Staples</p>
            <span class="text-gray-500 font-medium text-sm">(Use Menu to Shop)</span>
          </div>
        </div>
        
        <!-- Other Departments (Decorative) -->
        <div class="bg-white rounded-xl shadow-md hover:shadow-xl transition-all cursor-pointer overflow-hidden group border border-gray-100">
          <div class="h-48 overflow-hidden bg-gray-100">
            <img src="/images/Clothing.jpg" alt="Clothing" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
          </div>
          <div class="p-6">
             <h2 class="text-xl font-bold mb-2">Clothing</h2>
             <p class="text-gray-600 mb-4">Fashion for Men, Women & Kids</p>
             <span class="text-[#0071DC] font-medium group-hover:underline">Shop Now &rarr;</span>
          </div>
        </div>

      </div>
    </main>
    
    <PermissionModal />
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import PermissionModal from '../components/PermissionModal.vue'

export default {
  name: 'DEPARTMENTS',
  components: {
    PermissionModal
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleGoToElectronics = async () => {
      // FSM: ACT_DEPT_GO_TO_ELECTRONICS_DIRECT
      store.currentPageId = 'ELECTRONICS_CATEGORY'
      await router.push({ name: 'ELECTRONICS_CATEGORY' })
    }

    const handleGoToGrocery = async () => {
      // FSM: ACT_DEPT_GO_TO_GROCERY_MENU
      store.currentPageId = 'GROCERY_CATEGORY'
      await router.push({ name: 'GROCERY_CATEGORY' })
    }

    const handleGoToHome = async () => {
      // FSM: ACT_DEPT_GO_TO_HOME_DIRECT
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      handleGoToElectronics,
      handleGoToGrocery,
      handleGoToHome
    }
  }
}
</script>