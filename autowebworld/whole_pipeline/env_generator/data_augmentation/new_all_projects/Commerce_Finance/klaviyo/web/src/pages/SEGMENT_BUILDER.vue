<template>
  <div class="min-h-screen bg-slate-50 py-12">
    <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="bg-white shadow rounded-lg overflow-hidden">
        <div class="px-4 py-5 sm:p-6 space-y-8">
          <div class="flex items-center justify-between">
            <h2 class="text-2xl font-bold text-slate-900">Segment Definition</h2>
          </div>
          
          <!-- Name -->
          <div>
            <label for="segment-name-input" class="block text-sm font-medium text-slate-700">Name</label>
            <input 
              type="text" 
              id="segment-name-input"
              v-model="inputName"
              @input="handleNameInput"
              class="mt-1 shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full sm:text-sm border-slate-300 rounded-md py-2 px-3"
              placeholder="e.g. VIP Customers"
            />
          </div>

          <!-- Definition Area -->
          <div class="bg-slate-50 p-6 rounded-lg border border-slate-200">
            <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wide mb-4">Definition</h3>
            
            <div class="flex flex-col md:flex-row gap-4 items-center">
               <span class="text-slate-700 font-medium">If someone</span>
               
               <div class="relative w-full md:w-64">
                 <button 
                  id="condition-dropdown"
                  @click="toggleDropdown"
                  class="w-full bg-white border border-slate-300 rounded-lg py-2 px-3 flex items-center justify-between shadow-sm hover:border-blue-500 focus:outline-none"
                 >
                   <span class="block truncate">{{ selectedConditionLabel || 'Choose condition' }}</span>
                   <svg class="h-4 w-4 text-slate-400" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
                 </button>
                 
                 <div v-if="isOpen" class="absolute z-10 mt-1 w-full bg-white shadow-xl rounded-lg py-1 ring-1 ring-black ring-opacity-5">
                    <div 
                      id="condition-opened-email"
                      @click="selectCondition('opened_email')"
                      class="cursor-pointer px-4 py-2 hover:bg-blue-50"
                    >
                      Opened Email
                    </div>
                    <div 
                      id="condition-clicked-link"
                      @click="selectCondition('clicked_link')"
                      class="cursor-pointer px-4 py-2 hover:bg-blue-50"
                    >
                      Clicked Link
                    </div>
                 </div>
               </div>
               
               <span class="text-slate-700 font-medium">equals</span>
               
               <input 
                 type="text" 
                 id="condition-value-input"
                 v-model="inputValue"
                 @input="handleValueInput"
                 class="shadow-sm focus:ring-blue-500 focus:border-blue-500 block w-full md:w-48 sm:text-sm border-slate-300 rounded-md py-2 px-3"
                 placeholder="Value..."
               />
            </div>
          </div>

        </div>
        <div class="px-4 py-4 bg-slate-50 border-t border-slate-200 sm:px-6 flex justify-between">
          <button 
            id="back-lists-segments"
            @click="goBack"
            class="inline-flex justify-center py-2 px-4 border border-slate-300 shadow-sm text-sm font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
          >
            Cancel
          </button>
          <button 
            id="btn-save-segment"
            @click="saveSegment"
            :disabled="!isValid"
            class="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none disabled:opacity-50"
          >
            Create Segment
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SEGMENT_BUILDER',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const inputName = ref('')
    const inputValue = ref('')
    const isOpen = ref(false)

    const selectedConditionLabel = computed(() => {
      if (store.segment_condition_type === 'opened_email') return 'Opened Email'
      if (store.segment_condition_type === 'clicked_link') return 'Clicked Link'
      return ''
    })

    function handleNameInput() {
      store.segment_name = `Segment ${inputName.value}`
    }

    function handleValueInput() {
      store.segment_condition_value = inputValue.value
    }

    function toggleDropdown() {
      isOpen.value = !isOpen.value
    }

    function selectCondition(type) {
      store.segment_condition_type = type
      isOpen.value = false
    }

    const isValid = computed(() => {
      return store.segment_name && store.segment_name.length > 0 &&
             store.segment_condition_type && store.segment_condition_type.length > 0 &&
             store.segment_condition_value && store.segment_condition_value.length > 0
    })

    async function goBack() {
      store.setCurrentPageId('LISTS_SEGMENTS')
      await router.push({ name: 'LISTS_SEGMENTS' })
    }

    async function saveSegment() {
      if (!isValid.value) return
      store.setCurrentPageId('SEGMENT_CREATED_SUCCESS')
      await router.push({ name: 'SEGMENT_CREATED_SUCCESS' })
    }

    return {
      inputName,
      inputValue,
      isOpen,
      selectedConditionLabel,
      handleNameInput,
      handleValueInput,
      toggleDropdown,
      selectCondition,
      isValid,
      goBack,
      saveSegment
    }
  }
}
</script>