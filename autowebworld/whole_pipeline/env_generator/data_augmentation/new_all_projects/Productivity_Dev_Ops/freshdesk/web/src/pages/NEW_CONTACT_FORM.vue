<template>
  <div class="min-h-screen bg-slate-50 font-inter text-slate-900 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center">
         <button id="new-contact-back-contacts" @click="handleBack" class="mr-4 text-slate-500 hover:text-blue-600 transition-colors">
            ← Back
         </button>
         <h1 class="text-xl font-bold text-slate-900">New Contact</h1>
      </div>
    </header>

    <main class="flex-1 max-w-lg mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
       <div class="bg-white p-8 rounded-lg shadow-sm border border-slate-200 space-y-6">
          
          <!-- Name -->
          <div>
             <label for="new-contact-name" class="block text-sm font-medium text-slate-700 mb-1">Full Name</label>
             <input type="text" 
                    id="new-contact-name" 
                    v-model="name" 
                    @input="handleNameInput"
                    class="block w-full border-slate-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2.5" 
                    placeholder="John Doe">
          </div>

          <!-- Email -->
          <div>
             <label for="new-contact-email" class="block text-sm font-medium text-slate-700 mb-1">Email Address</label>
             <input type="text" 
                    id="new-contact-email" 
                    v-model="email" 
                    @input="handleEmailInput"
                    class="block w-full border-slate-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 sm:text-sm p-2.5" 
                    placeholder="john@example.com">
          </div>

          <!-- Segment -->
          <div class="relative">
             <label class="block text-sm font-medium text-slate-700 mb-1">Segment</label>
             <div class="relative">
                <button id="new-contact-segment-dropdown" @click="toggleDropdown" class="w-full bg-white border border-slate-300 rounded-md py-2 px-3 text-left shadow-sm focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 sm:text-sm">
                   <span class="block truncate">{{ segment || 'Select Segment' }}</span>
                   <span class="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none text-slate-400">▼</span>
                </button>
                <div v-if="dropdownOpen" class="absolute z-10 mt-1 w-full bg-white shadow-lg max-h-60 rounded-md py-1 text-base ring-1 ring-black ring-opacity-5 overflow-auto focus:outline-none sm:text-sm">
                   <div id="segment-vip" @click="handleSelectSegment('VIP')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">VIP</div>
                   <div id="segment-standard" @click="handleSelectSegment('Standard')" class="cursor-pointer select-none relative py-2 pl-3 pr-9 hover:bg-blue-50 text-slate-900">Standard</div>
                </div>
             </div>
          </div>

          <div class="pt-4 flex justify-end">
             <button id="btn-new-contact-review" 
                     @click="handleReview"
                     :disabled="!isValid"
                     :class="[
                        'px-6 py-2.5 rounded-md font-medium text-sm transition-colors shadow-sm',
                        isValid ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-slate-200 text-slate-400 cursor-not-allowed'
                     ]">
               Review Contact
             </button>
          </div>
       </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'NEW_CONTACT_FORM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()

    const name = ref('')
    const email = ref('')
    const segment = ref('VIP') // Default in FSM example action is VIP
    const dropdownOpen = ref(false)

    const isValid = computed(() => {
        return name.value.length > 0 && email.value.length > 0
    })

    const handleNameInput = () => {
        if (name.value) signatureStore.new_contact_name = 'has_name'
        else signatureStore.new_contact_name = null
    }

    const handleEmailInput = () => {
        if (email.value) signatureStore.new_contact_email = 'has_email'
        else signatureStore.new_contact_email = null
    }

    const toggleDropdown = () => dropdownOpen.value = !dropdownOpen.value

    const handleSelectSegment = (val) => {
        segment.value = val
        signatureStore.new_contact_segment = val
        dropdownOpen.value = false
    }

    const handleReview = async () => {
        if (!isValid.value) return
        signatureStore.setCurrentPageId('NEW_CONTACT_REVIEW')
        await router.push({ name: 'NEW_CONTACT_REVIEW' })
    }

    const handleBack = async () => {
        signatureStore.setCurrentPageId('CONTACTS_LIST')
        await router.push({ name: 'CONTACTS_LIST' })
    }

    return {
        name,
        email,
        segment,
        dropdownOpen,
        isValid,
        handleNameInput,
        handleEmailInput,
        toggleDropdown,
        handleSelectSegment,
        handleReview,
        handleBack
    }
  }
}
</script>