<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Nav -->
    <nav class="bg-white shadow-sm sticky top-0 z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <div id="header-logo-home" class="flex-shrink-0 flex items-center cursor-pointer" @click="goHome">
              <span class="text-2xl font-bold text-blue-700">Coursera</span>
            </div>
            <div class="ml-10 flex items-baseline space-x-4">
               <h1 class="text-xl font-semibold text-gray-800">Professional Certificates</h1>
            </div>
          </div>
        </div>
      </div>
    </nav>

    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Search Bar -->
      <div class="flex justify-center mb-8">
        <div class="w-full max-w-2xl relative">
          <input 
            id="pro-cert-search-input"
            type="text" 
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            placeholder="Search certificates" 
            class="w-full px-5 py-3 border border-gray-300 rounded-full shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
          >
          <button 
            @click="handleSearch"
            class="absolute right-2 top-2 bg-blue-700 text-white p-2 rounded-full hover:bg-blue-800"
          >
            <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </button>
        </div>
      </div>

      <!-- List -->
      <div id="pro-cert-results-list" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        <div 
          v-for="cert in filteredCerts" 
          :key="cert.id"
          :class="getCardClass(cert)"
          class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow cursor-pointer flex flex-col h-full"
          @click="openCert(cert)"
        >
          <div class="h-48 bg-gray-200 flex-shrink-0 relative">
            <img :src="cert.image" :alt="cert.title" class="w-full h-full object-cover">
            <div class="absolute top-2 right-2 bg-black/70 text-white px-2 py-1 rounded text-xs font-bold uppercase">
              {{ cert.provider }}
            </div>
          </div>
          <div class="p-6 flex-1 flex flex-col justify-between">
            <div>
              <div class="flex items-center text-xs font-semibold tracking-wide uppercase text-blue-600 mb-1">
                Professional Certificate
              </div>
              <h3 class="text-xl font-bold text-gray-900 mb-2">{{ cert.title }}</h3>
              <p class="text-gray-600 text-sm mb-4 line-clamp-2">{{ cert.description }}</p>
              
              <div class="flex items-center space-x-4 text-sm text-gray-500">
                <span class="flex items-center">
                  <span class="text-yellow-400 mr-1">★</span> {{ cert.rating }}
                </span>
                <span>{{ cert.duration }} months</span>
              </div>
            </div>
            
            <div class="mt-4 flex items-center justify-between">
                <span class="text-blue-700 font-bold">${{ cert.price }}</span>
                <span class="text-xs text-gray-400">Flexible schedule</span>
            </div>
          </div>
          
           <!-- Hidden data attributes for testing -->
          <div :class="`data-id-${cert.id}`" class="hidden"></div>
        </div>
        
        <div v-if="filteredCerts.length === 0" class="col-span-full text-center py-12">
            <p class="text-gray-500 text-lg">No certificates found.</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PROFESSIONAL_CERT_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')

    const filteredCerts = computed(() => {
      let result = [...dataStore.professional_certs]

      if (searchQuery.value) {
         const q = searchQuery.value.toLowerCase()
         result = result.filter(c => 
           c.title.toLowerCase().includes(q) || 
           c.provider.toLowerCase().includes(q)
         )
      }

      return result
    })

    function handleSearch() {
      store.pro_cert_list_has_searched = true
      const match = filteredCerts.value.length > 0 ? filteredCerts.value[0] : null
      store.matched_pro_cert_id = match ? match.id : null
    }

    function getCardClass(cert) {
      if (store.pro_cert_list_has_searched && cert.id === store.matched_pro_cert_id) {
        return 'pro-cert-card-matched'
      }
      return 'pro-cert-card-visible'
    }

    async function openCert(cert) {
      store.selected_pro_cert_id = cert.id
      store.setCurrentPageId('PROFESSIONAL_CERT_DETAIL')
      await router.push({ name: 'PROFESSIONAL_CERT_DETAIL', params: { id: cert.id } })
    }

    async function goHome() {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      searchQuery,
      filteredCerts,
      handleSearch,
      getCardClass,
      openCert,
      goHome
    }
  }
}
</script>