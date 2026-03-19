<template>
  <div class="min-h-screen bg-slate-50 font-inter text-slate-900 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <h1 class="text-xl font-bold text-slate-900">Review Contact</h1>
      </div>
    </header>

    <main class="flex-1 max-w-lg mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
       <div class="bg-white p-8 rounded-lg shadow-sm border border-slate-200 space-y-6">
          <div class="border-b border-slate-100 pb-4">
             <h3 class="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Name</h3>
             <p class="text-lg font-medium text-slate-900">{{ signatureStore.new_contact_name === 'has_name' ? 'Simulated Name' : 'No Name' }}</p>
          </div>
          <div class="border-b border-slate-100 pb-4">
             <h3 class="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Email</h3>
             <p class="text-base text-slate-800">{{ signatureStore.new_contact_email === 'has_email' ? 'simulated@example.com' : 'No Email' }}</p>
          </div>
          <div>
             <h3 class="text-sm font-medium text-slate-500 uppercase tracking-wider mb-1">Segment</h3>
             <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
               {{ signatureStore.new_contact_segment || 'VIP' }}
             </span>
          </div>

          <div class="flex justify-between pt-4">
             <button id="new-contact-review-back-form" 
                     @click="handleBack" 
                     class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-2 px-4 rounded-md shadow-sm transition-colors">
                Edit
             </button>
             <button id="btn-submit-new-contact" 
                     @click="handleSubmit" 
                     class="bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded-md shadow-sm transition-colors">
                Create Contact
             </button>
          </div>
       </div>
    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'NEW_CONTACT_REVIEW',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const handleBack = async () => {
        signatureStore.setCurrentPageId('NEW_CONTACT_FORM')
        await router.push({ name: 'NEW_CONTACT_FORM' })
    }

    const handleSubmit = async () => {
        // Create actual contact in mock data
        const newContact = {
            id: `c${Date.now()}`,
            name: 'Simulated Name',
            email: 'simulated@example.com',
            segment: signatureStore.new_contact_segment || 'VIP',
            avatar: '/images/photo1765352758.jpg' // Default image
        }
        dataStore.addContact(newContact)

        signatureStore.created_contact_id = newContact.id
        signatureStore.setCurrentPageId('CONTACT_CREATION_SUCCESS')
        await router.push({ name: 'CONTACT_CREATION_SUCCESS' })
    }

    return {
        signatureStore,
        handleBack,
        handleSubmit
    }
  }
}
</script>