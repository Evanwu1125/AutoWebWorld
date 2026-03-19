<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-6">
    <div class="bg-white rounded-xl shadow-2xl max-w-4xl w-full flex flex-col md:flex-row overflow-hidden h-[80vh]">
      
      <!-- Record Info Panel -->
      <div class="flex-1 p-8 overflow-y-auto">
         <div class="flex justify-between items-start mb-6">
           <div>
             <span class="inline-block px-2 py-0.5 rounded text-xs font-semibold bg-blue-100 text-blue-700 mb-2">
               {{ record?.status || 'No Status' }}
             </span>
             <h1 class="text-3xl font-bold text-gray-900">{{ record?.title || 'Untitled Record' }}</h1>
           </div>
           
           <button id="back-grid-view" @click="goBack" class="p-2 hover:bg-gray-100 rounded-full transition-colors">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
             </svg>
           </button>
         </div>

         <div v-if="record?.image" class="mb-6 rounded-lg overflow-hidden h-64 shadow-md">
           <img :src="record.image" class="w-full h-full object-cover" />
         </div>

         <div class="space-y-6">
           <div class="grid grid-cols-2 gap-4">
             <div>
               <label class="block text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Due Date</label>
               <div class="text-gray-900 font-medium">{{ record?.due_date }}</div>
             </div>
             <div>
               <label class="block text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Priority</label>
               <div class="text-gray-900 font-medium">{{ record?.priority }}</div>
             </div>
             <div>
               <label class="block text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Assignee</label>
               <div class="flex items-center gap-2">
                 <div class="w-6 h-6 rounded-full bg-gray-200 flex items-center justify-center text-xs">A</div>
                 <span class="text-gray-900 font-medium">{{ record?.assigned_to }}</span>
               </div>
             </div>
           </div>

           <div>
             <label class="block text-xs font-bold text-gray-400 uppercase tracking-wider mb-1">Description</label>
             <p class="text-gray-600 leading-relaxed">
               This is a detailed view of the record. You can see all fields here, view activity history, and comment on the record.
             </p>
           </div>
         </div>
      </div>

      <!-- Action Sidebar -->
      <div class="w-full md:w-80 bg-gray-50 border-l border-gray-200 p-6 flex flex-col gap-4">
         <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Actions</h3>
         
         <button 
           id="edit-record-button" 
           @click="openEdit" 
           class="w-full bg-white border border-gray-300 hover:border-blue-500 hover:text-blue-600 text-gray-700 font-semibold py-2 px-4 rounded-lg shadow-sm transition-all flex items-center justify-center gap-2"
         >
           <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
           </svg>
           Edit Record
         </button>

         <button 
           id="open-form-view" 
           @click="openFormView" 
           class="w-full bg-white border border-gray-300 hover:border-purple-500 hover:text-purple-600 text-gray-700 font-semibold py-2 px-4 rounded-lg shadow-sm transition-all flex items-center justify-center gap-2"
         >
           <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
           </svg>
           Share Form View
         </button>
         
         <div class="mt-auto">
            <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-2">Activity</h3>
            <div class="space-y-3">
               <div class="flex gap-2 text-sm">
                  <div class="w-6 h-6 rounded-full bg-blue-100 flex-shrink-0"></div>
                  <div>
                     <span class="font-bold">You</span> created this record
                     <div class="text-xs text-gray-400">Just now</div>
                  </div>
               </div>
            </div>
         </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'RECORD_DETAIL',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const record = computed(() => {
      return dataStore.records.find(r => r.id === store.selected_record_id)
    })

    const goBack = async () => {
      // Default to grid view, could be kanban depending on history, but FSM says TABLE_GRID_VIEW
      store.setCurrentPageId('TABLE_GRID_VIEW')
      await router.push({ name: 'TABLE_GRID_VIEW' })
    }

    const openEdit = async () => {
      store.setCurrentPageId('RECORD_EDIT_FORM')
      await router.push({ name: 'RECORD_EDIT_FORM' })
    }

    const openFormView = async () => {
      store.setCurrentPageId('FORM_VIEW_SUBMISSION')
      await router.push({ name: 'FORM_VIEW_SUBMISSION' })
    }

    return {
      record,
      goBack,
      openEdit,
      openFormView
    }
  }
}
</script>