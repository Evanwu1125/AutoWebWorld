<template>
  <div class="min-h-screen bg-slate-50 font-inter text-slate-900 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="back-contacts-list" @click="handleBackToList" class="mr-4 text-slate-500 hover:text-blue-600 transition-colors">
            ← Back
         </button>
         <h1 class="text-xl font-bold text-slate-900">{{ contact?.name }}</h1>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full flex flex-col lg:flex-row gap-6">
       <!-- Sidebar: Contact Info -->
       <div class="w-full lg:w-80 space-y-6">
          <div class="bg-white p-6 rounded-lg shadow-sm border border-slate-200 text-center">
             <img :src="contact?.avatar" class="h-32 w-32 rounded-full mx-auto mb-4 border-4 border-slate-100" alt="Avatar">
             <h2 class="text-lg font-bold text-slate-900">{{ contact?.name }}</h2>
             <p class="text-slate-500 text-sm mb-4">{{ contact?.segment }}</p>
             
             <div class="text-left space-y-3 pt-4 border-t border-slate-100">
                <div>
                   <label class="text-xs font-semibold text-slate-400 uppercase">Email</label>
                   <p class="text-sm text-slate-900">{{ contact?.email }}</p>
                </div>
                <div>
                   <label class="text-xs font-semibold text-slate-400 uppercase">Phone</label>
                   <p class="text-sm text-slate-900">{{ contact?.phone || 'Not provided' }}</p>
                </div>
             </div>
          </div>
       </div>

       <!-- Main Content: Activity/Tickets -->
       <div class="flex-1 space-y-6">
          <div class="bg-white p-6 rounded-lg shadow-sm border border-slate-200 min-h-[400px]">
             <h3 class="text-lg font-medium text-slate-900 mb-6 border-b border-slate-100 pb-2">Recent Tickets</h3>
             <!-- Mock list of tickets for this contact -->
             <div class="space-y-4">
                <div class="p-4 border border-slate-100 rounded-md hover:bg-slate-50 transition-colors">
                   <div class="flex justify-between items-start">
                      <div>
                         <p class="text-sm font-medium text-blue-600">#t1024 - Login issue</p>
                         <p class="text-xs text-slate-500 mt-1">Created 2 days ago</p>
                      </div>
                      <span class="px-2 py-0.5 rounded text-xs bg-green-100 text-green-800">Open</span>
                   </div>
                </div>
                <div class="p-4 border border-slate-100 rounded-md hover:bg-slate-50 transition-colors">
                   <div class="flex justify-between items-start">
                      <div>
                         <p class="text-sm font-medium text-slate-700">#t985 - Billing question</p>
                         <p class="text-xs text-slate-500 mt-1">Created 1 week ago</p>
                      </div>
                      <span class="px-2 py-0.5 rounded text-xs bg-gray-100 text-gray-800">Closed</span>
                   </div>
                </div>
             </div>
          </div>
       </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CONTACT_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const contactId = route.params.id
    const contact = computed(() => dataStore.getContactById(contactId))

    const handleBackToList = async () => {
        signatureStore.setCurrentPageId('CONTACTS_LIST')
        await router.push({ name: 'CONTACTS_LIST' })
    }

    return {
        contact,
        handleBackToList
    }
  }
}
</script>