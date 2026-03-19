<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-6">
    <div class="bg-white rounded-xl shadow-xl w-full max-w-2xl overflow-hidden flex flex-col md:flex-row h-auto md:h-[600px]">
      
      <!-- Preview Side -->
      <div class="w-full md:w-1/2 bg-gray-100 p-8 flex flex-col items-center justify-center border-r border-gray-200">
        <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-8">Preview</h3>
        
        <div class="w-40 h-40 bg-white rounded-2xl shadow-lg flex items-center justify-center mb-6 relative overflow-hidden transition-all duration-300 transform hover:scale-105">
           <!-- Dynamic Color Background -->
           <div :class="`absolute inset-0 opacity-10 bg-${selectedColor}-500`"></div>
           
           <!-- Dynamic Icon -->
           <div :class="`text-5xl text-${selectedColor}-600`">
             <span v-if="selectedIcon === 'grid'">⊞</span>
             <span v-else-if="selectedIcon === 'calendar'">📅</span>
             <span v-else-if="selectedIcon === 'kanban'">📋</span>
             <span v-else>📄</span>
           </div>
        </div>
        
        <h2 class="text-2xl font-bold text-gray-800 text-center">{{ baseName || 'Untitled Base' }}</h2>
      </div>

      <!-- Form Side -->
      <div class="w-full md:w-1/2 p-8 flex flex-col">
         <div class="flex justify-between items-center mb-8">
           <h1 class="text-2xl font-bold text-gray-900">Create New Base</h1>
           <button id="back-bases-dashboard" @click="goBack" class="text-gray-400 hover:text-gray-600">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
             </svg>
           </button>
         </div>

         <div class="space-y-6 flex-1">
           <!-- Name Input -->
           <div>
             <label class="block text-sm font-medium text-gray-700 mb-2">Base Name</label>
             <input 
               id="base-name-input"
               v-model="baseName"
               @input="handleNameInput"
               type="text" 
               class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
               placeholder="e.g., Marketing Q4"
             >
           </div>

           <!-- Color Picker -->
           <div class="relative">
             <label class="block text-sm font-medium text-gray-700 mb-2">Color</label>
             <button 
               id="base-color-dropdown"
               @click="colorDropdownOpen = !colorDropdownOpen"
               class="w-full flex items-center justify-between px-4 py-2 border border-gray-300 rounded-lg bg-white hover:border-blue-400 transition-colors"
             >
               <div class="flex items-center gap-2">
                 <div :class="`w-4 h-4 rounded-full bg-${selectedColor}-500`"></div>
                 <span class="capitalize">{{ selectedColor }}</span>
               </div>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </button>
             
             <div v-if="colorDropdownOpen" class="absolute top-full left-0 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-xl z-20 p-2 grid grid-cols-3 gap-2">
               <div id="base-color-blue" @click="selectColor('blue')" class="cursor-pointer hover:bg-gray-100 p-2 rounded flex items-center gap-2">
                 <div class="w-4 h-4 rounded-full bg-blue-500"></div> Blue
               </div>
               <div id="base-color-green" @click="selectColor('green')" class="cursor-pointer hover:bg-gray-100 p-2 rounded flex items-center gap-2">
                 <div class="w-4 h-4 rounded-full bg-green-500"></div> Green
               </div>
               <div id="base-color-red" @click="selectColor('red')" class="cursor-pointer hover:bg-gray-100 p-2 rounded flex items-center gap-2">
                 <div class="w-4 h-4 rounded-full bg-red-500"></div> Red
               </div>
             </div>
           </div>

           <!-- Icon Picker (Hover Menu) -->
           <div class="relative group" id="base-icon-menu" @mouseenter="iconMenuOpen = true" @mouseleave="iconMenuOpen = false">
             <label class="block text-sm font-medium text-gray-700 mb-2">Icon</label>
             <div class="w-full flex items-center justify-between px-4 py-2 border border-gray-300 rounded-lg bg-white hover:border-blue-400 transition-colors cursor-pointer">
               <div class="flex items-center gap-2">
                 <span class="text-lg">
                   <span v-if="selectedIcon === 'grid'">⊞</span>
                   <span v-else-if="selectedIcon === 'calendar'">📅</span>
                   <span v-else-if="selectedIcon === 'kanban'">📋</span>
                 </span>
                 <span class="capitalize">{{ selectedIcon }}</span>
               </div>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </div>
             
             <div v-if="iconMenuOpen" class="absolute top-full left-0 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-xl z-20">
               <div class="icon-grid p-3 hover:bg-gray-50 cursor-pointer flex items-center gap-3" @click="selectIcon('grid')">
                 <span class="text-xl">⊞</span> Grid
               </div>
               <div class="icon-calendar p-3 hover:bg-gray-50 cursor-pointer flex items-center gap-3" @click="selectIcon('calendar')">
                 <span class="text-xl">📅</span> Calendar
               </div>
               <div class="icon-kanban p-3 hover:bg-gray-50 cursor-pointer flex items-center gap-3" @click="selectIcon('kanban')">
                 <span class="text-xl">📋</span> Kanban
               </div>
             </div>
           </div>

           <!-- Template Selection (Final Step) -->
           <div class="relative pt-4">
             <label class="block text-sm font-medium text-gray-700 mb-2">Start with a Template</label>
             <button 
               id="template-dropdown"
               @click="templateDropdownOpen = !templateDropdownOpen"
               class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg shadow-md transition-all flex justify-between items-center"
               :disabled="!baseName"
               :class="{'opacity-50 cursor-not-allowed': !baseName}"
             >
               <span>Create Base</span>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </button>
             
             <div v-if="templateDropdownOpen" class="absolute bottom-full left-0 w-full mb-1 bg-white border border-gray-200 rounded-lg shadow-2xl z-30 overflow-hidden">
               <div class="p-2 bg-gray-50 text-xs font-bold text-gray-500 uppercase">Choose Template</div>
               <div id="template-project-tracker" @click="createBase('project-tracker')" class="p-3 hover:bg-blue-50 cursor-pointer border-b border-gray-100">
                 <div class="font-bold text-gray-800">Project Tracker</div>
                 <div class="text-xs text-gray-500">Track tasks and deadlines</div>
               </div>
               <div id="template-sales-crm" @click="createBase('sales-crm')" class="p-3 hover:bg-blue-50 cursor-pointer border-b border-gray-100">
                 <div class="font-bold text-gray-800">Sales CRM</div>
                 <div class="text-xs text-gray-500">Manage leads and deals</div>
               </div>
               <div id="template-content-calendar" @click="createBase('content-calendar')" class="p-3 hover:bg-blue-50 cursor-pointer">
                 <div class="font-bold text-gray-800">Content Calendar</div>
                 <div class="text-xs text-gray-500">Schedule posts and content</div>
               </div>
             </div>
           </div>
         </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'BASE_CREATE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const baseName = ref('')
    const selectedColor = ref('blue')
    const selectedIcon = ref('grid')
    
    const colorDropdownOpen = ref(false)
    const iconMenuOpen = ref(false)
    const templateDropdownOpen = ref(false)

    const handleNameInput = () => {
      // ACT_BASE_CREATE_TYPE_NAME
      store.base_name_input = baseName.value
    }

    const selectColor = (color) => {
      // ACT_BASE_CREATE_SELECT_COLOR
      selectedColor.value = color
      store.base_color = color
      colorDropdownOpen.value = false
    }

    const selectIcon = (icon) => {
      // ACT_BASE_CREATE_SELECT_ICON
      selectedIcon.value = icon
      store.base_icon = icon
      iconMenuOpen.value = false
    }

    const createBase = async (template) => {
      // ACT_BASE_CREATE_SELECT_TEMPLATE
      store.template_choice = template
      
      // Effect: Append to bases (mock implementation)
      const newId = 'base_' + Date.now()
      const newBase = {
        id: newId,
        name: store.base_name_input,
        color: store.base_color,
        icon: store.base_icon,
        starred: false,
        activity: 0,
        last_viewed: new Date().toISOString(),
        image: '/images/Base.jpg' // Placeholder
      }
      
      store.bases.push(newBase)
      store.created_base_id = newId
      
      store.setCurrentPageId('BASE_CREATED_SUCCESS')
      await router.push({ name: 'BASE_CREATED_SUCCESS' })
    }

    const goBack = async () => {
      store.setCurrentPageId('BASES_DASHBOARD')
      await router.push({ name: 'BASES_DASHBOARD' })
    }

    return {
      baseName,
      selectedColor,
      selectedIcon,
      colorDropdownOpen,
      iconMenuOpen,
      templateDropdownOpen,
      
      handleNameInput,
      selectColor,
      selectIcon,
      createBase,
      goBack
    }
  }
}
</script>