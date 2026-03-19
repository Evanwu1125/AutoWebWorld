<template>
  <div class="home-page min-h-screen flex flex-col">
    <!-- Header -->
    <header class="bg-[#0071DC] text-white p-4 sticky top-0 z-50 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center justify-between gap-4">
        <div id="logo-home" class="font-bold text-xl tracking-tight flex items-center gap-2 cursor-pointer">
          <div class="w-8 h-8 bg-[#FFC220] rounded-full flex items-center justify-center text-[#0071DC]">
            <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" /></svg>
          </div>
          Walmart
        </div>
        
        <!-- Search Bar (Visual Only) -->
        <div class="flex-1 max-w-2xl mx-4 hidden md:block">
          <div class="relative">
            <input type="text" placeholder="Search everything at Walmart online and in store" class="w-full px-5 py-2.5 rounded-full text-gray-900 focus:outline-none focus:ring-2 focus:ring-[#FFC220]" />
            <div class="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 bg-[#FFC220] rounded-full flex items-center justify-center text-gray-900">
               <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
            </div>
          </div>
        </div>

        <!-- Navigation Actions -->
        <div class="flex items-center gap-6">
          
          <!-- Account Link -->
          <div 
            id="nav-account-direct" 
            @click="handleGoToAccount"
            class="flex items-center gap-2 cursor-pointer hover:bg-white/10 px-3 py-2 rounded-full transition-colors"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
            <div class="hidden sm:block text-sm font-medium">Account</div>
          </div>

          <!-- Cart Link (Hover Action) -->
          <div 
            id="header-account" 
            class="relative group py-2"
          >
            <div class="flex items-center gap-2 cursor-pointer text-white">
               <div class="flex flex-col items-center">
                 <span class="text-xs font-light">Reorder</span>
                 <span class="font-bold text-sm">My Items</span>
               </div>
            </div>
            
            <!-- Hover Dropdown content that contains the cart link -->
            <div class="absolute right-0 top-full pt-2 w-48 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none group-hover:pointer-events-auto z-50">
               <div class="bg-white rounded-lg shadow-xl p-4 text-gray-900">
                  <div class="font-bold mb-2 border-b pb-2">Your Lists</div>
                  <div 
                    id="header-cart-link"
                    @click="handleGoToCart"
                    class="block w-full text-left px-2 py-1 hover:bg-gray-100 rounded cursor-pointer text-blue-600 font-medium"
                  >
                    View Cart
                  </div>
               </div>
            </div>
          </div>
          
          <!-- Direct Cart Icon -->
          <div class="relative cursor-pointer">
             <svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" /></svg>
             <span class="absolute -top-1 -right-1 bg-[#FFC220] text-black text-xs font-bold w-5 h-5 rounded-full flex items-center justify-center">0</span>
          </div>

        </div>
      </div>
      
      <!-- Sub Header Navigation -->
      <div class="max-w-7xl mx-auto mt-2 flex items-center gap-6 text-sm font-medium border-t border-white/20 pt-2">
        <!-- Departments Menu Toggle -->
        <div id="menu-toggle" class="flex items-center gap-2 cursor-pointer hover:underline">
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" /></svg>
          Departments
        </div>
        
        <!-- Dropdown Menu Item (Hidden by default, shown via CSS logic usually, but here simulating structure) -->
        <!-- For FSM: click #menu-toggle -> click #menu-item-departments -->
        <!-- Implementation: Toggle state ref locally -->
        
        <div 
          id="nav-departments-direct" 
          @click="handleGoToDepartments"
          class="cursor-pointer hover:underline"
        >
          Deals
        </div>
        <div class="cursor-pointer hover:underline">Grocery</div>
        <div class="cursor-pointer hover:underline">Electronics</div>
      </div>
    </header>
    
    <!-- Mega Menu (Conditional) -->
    <div v-if="showMenu" class="fixed inset-0 z-40 bg-black/50" @click="showMenu = false">
      <div class="bg-white w-64 h-full pt-20 p-4 shadow-xl" @click.stop>
        <div class="font-bold text-lg mb-4">All Departments</div>
        <div 
          id="menu-item-departments" 
          @click="handleGoToDepartmentsFromMenu"
          class="p-3 hover:bg-gray-100 rounded cursor-pointer flex justify-between items-center"
        >
          Browse All
          <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" /></svg>
        </div>
      </div>
    </div>

    <!-- Main Content -->
    <main class="flex-1 bg-[#F3F4F6]">
      <!-- Hero Section -->
      <div class="bg-[#E6F1FC] py-8 px-4">
        <div class="max-w-7xl mx-auto grid md:grid-cols-2 gap-8 items-center rounded-2xl overflow-hidden bg-white shadow-lg">
           <div class="p-8 md:p-12">
             <h1 class="text-4xl md:text-5xl font-extrabold text-[#2E2F32] mb-4 leading-tight">
               Fresh savings <br/>
               <span class="text-[#0071DC]">delivered to you.</span>
             </h1>
             <p class="text-lg text-gray-600 mb-8">Get up to 50% off on fresh groceries and summer essentials.</p>
             <button 
               @click="handleGoToDepartments"
               class="bg-[#0071DC] text-white px-8 py-3 rounded-full font-bold shadow-lg hover:bg-[#005bb5] transition-all transform hover:-translate-y-1"
             >
               Shop Now
             </button>
           </div>
           <div class="h-64 md:h-full min-h-[400px] relative">
             <img src="/images/Shopping.jpg" alt="Family Shopping" class="absolute inset-0 w-full h-full object-cover" />
           </div>
        </div>
      </div>
      
      <!-- Featured Categories -->
      <div class="max-w-7xl mx-auto py-12 px-4">
        <h2 class="text-2xl font-bold mb-6">Featured Departments</h2>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
           <!-- Electronics -->
           <div @click="handleGoToDepartments" class="group cursor-pointer bg-white p-4 rounded-xl shadow-sm hover:shadow-md transition-all">
             <div class="aspect-square bg-gray-100 rounded-lg mb-3 overflow-hidden">
               <img src="/images/Electronics.jpg" alt="Electronics" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
             </div>
             <h3 class="font-semibold">Electronics</h3>
           </div>
           <!-- Grocery -->
           <div @click="handleGoToDepartments" class="group cursor-pointer bg-white p-4 rounded-xl shadow-sm hover:shadow-md transition-all">
             <div class="aspect-square bg-gray-100 rounded-lg mb-3 overflow-hidden">
               <img src="/images/Grocery.jpg" alt="Grocery" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
             </div>
             <h3 class="font-semibold">Grocery</h3>
           </div>
           <!-- Home -->
           <div class="group cursor-pointer bg-white p-4 rounded-xl shadow-sm hover:shadow-md transition-all">
             <div class="aspect-square bg-gray-100 rounded-lg mb-3 overflow-hidden">
               <img src="/images/HomeGarden.jpg" alt="Home" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
             </div>
             <h3 class="font-semibold">Home & Garden</h3>
           </div>
           <!-- Fashion -->
           <div class="group cursor-pointer bg-white p-4 rounded-xl shadow-sm hover:shadow-md transition-all">
             <div class="aspect-square bg-gray-100 rounded-lg mb-3 overflow-hidden">
               <img src="/images/Fashion.jpg" alt="Fashion" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
             </div>
             <h3 class="font-semibold">Fashion</h3>
           </div>
        </div>
      </div>
    </main>
    
    <CookieConsentModal />
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import CookieConsentModal from '../components/CookieConsentModal.vue'

export default {
  name: 'HOME',
  components: {
    CookieConsentModal
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const showMenu = ref(false)

    // Setup menu toggle logic for FSM
    onMounted(() => {
      // Attach click listener for #menu-toggle to toggle state
      const toggleBtn = document.getElementById('menu-toggle')
      if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
          showMenu.value = !showMenu.value
        })
      }
    })

    const handleGoToDepartments = async () => {
      // FSM: ACT_HOME_GO_TO_DEPARTMENTS_DIRECT
      store.currentPageId = 'DEPARTMENTS'
      await router.push({ name: 'DEPARTMENTS' })
    }
    
    const handleGoToDepartmentsFromMenu = async () => {
      // FSM: ACT_HOME_GO_TO_DEPARTMENTS_MENU
      store.currentPageId = 'DEPARTMENTS'
      await router.push({ name: 'DEPARTMENTS' })
    }

    const handleGoToCart = async () => {
      // FSM: ACT_HOME_GO_TO_CART_HOVER
      store.currentPageId = 'CART'
      await router.push({ name: 'CART' })
    }

    const handleGoToAccount = async () => {
      // FSM: ACT_HOME_GO_TO_ACCOUNT_DIRECT
      store.currentPageId = 'ACCOUNT'
      await router.push({ name: 'ACCOUNT' })
    }

    return {
      showMenu,
      handleGoToDepartments,
      handleGoToDepartmentsFromMenu,
      handleGoToCart,
      handleGoToAccount
    }
  }
}
</script>