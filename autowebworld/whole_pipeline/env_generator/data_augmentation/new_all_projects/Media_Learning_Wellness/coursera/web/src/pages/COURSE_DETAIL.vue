<template>
  <div class="min-h-screen bg-gray-50 pb-12">
    <!-- Breadcrumbs -->
    <div class="bg-white border-b border-gray-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div 
          id="breadcrumb-back-to-catalog" 
          class="flex items-center text-sm text-gray-500 hover:text-gray-700 cursor-pointer w-fit"
          @click="goBack"
        >
          <svg class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
          Back to Catalog
        </div>
      </div>
    </div>

    <div v-if="course" class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
      <!-- Course Header -->
      <div class="bg-white rounded-xl shadow-sm overflow-hidden flex flex-col lg:flex-row">
        <div class="lg:w-2/3 p-8">
          <div class="flex items-center space-x-2 text-sm text-blue-600 font-semibold uppercase tracking-wide mb-2">
            <span>{{ course.university }}</span>
            <span>•</span>
            <span>{{ course.level }}</span>
          </div>
          
          <h1 class="text-4xl font-extrabold text-gray-900 mb-4">{{ course.title }}</h1>
          <p class="text-xl text-gray-600 mb-6">{{ course.description }}</p>
          
          <div class="flex items-center space-x-6 mb-8">
            <div class="flex items-center">
              <span class="text-yellow-400 text-xl mr-1">★</span>
              <span class="font-bold text-gray-900">{{ course.rating }}</span>
              <span class="text-gray-500 ml-1">rating</span>
            </div>
            <div class="flex items-center">
              <span class="font-bold text-gray-900">{{ (course.students / 1000).toFixed(0) }}k</span>
              <span class="text-gray-500 ml-1">students</span>
            </div>
            <div class="flex items-center">
              <span class="font-bold text-gray-900">{{ course.duration }}</span>
              <span class="text-gray-500 ml-1">hours</span>
            </div>
          </div>

          <div class="flex items-center space-x-4">
            <div class="flex items-center">
              <img 
                src="https://randomuser.me/api/portraits/men/32.jpg" 
                alt="Instructor" 
                class="h-10 w-10 rounded-full"
              >
              <div class="ml-3">
                <p class="text-sm font-medium text-gray-900">{{ course.instructor }}</p>
                <p class="text-xs text-gray-500">Instructor</p>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Enrollment Card (Sidebar on desktop) -->
        <div class="lg:w-1/3 bg-gray-50 p-8 border-l border-gray-100 flex flex-col justify-center">
          <!-- Enroll Button with Hover Menu -->
          <div 
            id="enroll-button-menu"
            class="relative group w-full"
            @mouseenter="openEnrollMenu"
            @mouseleave="closeEnrollMenu"
          >
            <button class="w-full bg-blue-700 hover:bg-blue-800 text-white font-bold py-4 px-6 rounded-lg shadow-md transition-colors flex justify-center items-center">
              <span>Enroll for Free</span>
              <svg class="ml-2 h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            <!-- Hover Options -->
            <div v-if="isEnrollMenuOpen" class="absolute left-0 right-0 mt-2 bg-white rounded-md shadow-lg ring-1 ring-black ring-opacity-5 z-10">
              <div class="py-1">
                <div 
                  id="enroll-menu-purchase"
                  @click="selectEnrollType('purchase')"
                  class="block px-4 py-3 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                >
                  <div class="font-bold">Purchase Course</div>
                  <div class="text-xs text-gray-500">Get a certificate</div>
                </div>
                <div 
                  id="enroll-menu-audit"
                  @click="selectEnrollType('audit')"
                  class="block px-4 py-3 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                >
                  <div class="font-bold">Audit Only</div>
                  <div class="text-xs text-gray-500">Access materials for free</div>
                </div>
              </div>
            </div>
          </div>
          
          <p class="text-xs text-gray-500 text-center mt-4">
            Starts today. Financial aid available.
          </p>

          <!-- Conditional Buttons based on selection -->
          <div v-if="store.intended_enrollment_type === 'purchase'" class="mt-4">
            <button 
              id="enroll-now-primary"
              @click="goToEnrollFlow"
              class="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded transition-colors"
            >
              Continue to Purchase
            </button>
          </div>

          <div v-if="store.intended_enrollment_type === 'audit'" class="mt-4">
            <button 
              id="audit-this-course-link"
              @click="goToAuditFlow"
              class="w-full text-blue-700 hover:text-blue-800 font-medium py-2 px-4 rounded transition-colors"
            >
              Start Auditing
            </button>
          </div>
        </div>
      </div>

      <!-- Syllabus Tab Section -->
      <div class="mt-8 bg-white rounded-xl shadow-sm overflow-hidden p-6">
        <div class="border-b border-gray-200">
          <nav class="-mb-px flex space-x-8">
            <a href="#" class="border-blue-500 text-blue-600 whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm">
              About
            </a>
            <div 
              id="tab-syllabus"
              @click="viewSyllabus"
              class="border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm cursor-pointer"
            >
              Syllabus
            </div>
            <a href="#" class="border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm">
              Reviews
            </a>
          </nav>
        </div>
        <div class="py-6">
          <h3 class="text-lg font-medium text-gray-900">What you'll learn</h3>
          <ul class="mt-4 space-y-2 list-disc list-inside text-gray-600">
             <li>Master core concepts effectively</li>
             <li>Apply skills to real-world projects</li>
             <li>Gain industry-relevant knowledge</li>
          </ul>
        </div>
      </div>

    </div>
    
    <div v-else class="text-center py-12">
      Loading course details...
    </div>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'COURSE_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const courseId = route.params.id || store.selected_course_id
    const course = computed(() => dataStore.courses.find(c => c.id === courseId))

    const isEnrollMenuOpen = ref(false)

    function openEnrollMenu() {
      isEnrollMenuOpen.value = true
    }

    function closeEnrollMenu() {
      isEnrollMenuOpen.value = false
    }

    function selectEnrollType(type) {
      store.intended_enrollment_type = type
      closeEnrollMenu()
    }

    async function goToEnrollFlow() {
      if (store.intended_enrollment_type === 'purchase') {
        store.setCurrentPageId('ENROLLMENT_OPTIONS')
        await router.push({ name: 'ENROLLMENT_OPTIONS', params: { id: courseId } })
      }
    }

    async function goToAuditFlow() {
      if (store.intended_enrollment_type === 'audit') {
        store.setCurrentPageId('AUDIT_CONFIRM')
        await router.push({ name: 'AUDIT_CONFIRM', params: { id: courseId } })
      }
    }

    async function viewSyllabus() {
      store.setCurrentPageId('COURSE_SYLLABUS')
      await router.push({ name: 'COURSE_SYLLABUS', params: { id: courseId } })
    }

    async function goBack() {
      store.setCurrentPageId('COURSE_DISCOVERY')
      await router.push({ name: 'COURSE_DISCOVERY' })
    }

    return {
      store,
      course,
      isEnrollMenuOpen,
      openEnrollMenu,
      closeEnrollMenu,
      selectEnrollType,
      goToEnrollFlow,
      goToAuditFlow,
      viewSyllabus,
      goBack
    }
  }
}
</script>