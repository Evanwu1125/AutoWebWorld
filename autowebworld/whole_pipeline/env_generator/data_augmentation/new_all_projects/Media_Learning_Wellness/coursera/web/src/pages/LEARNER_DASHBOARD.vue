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
               <span class="text-lg font-medium text-gray-900">My Learning</span>
            </div>
          </div>
        </div>
      </div>
    </nav>

    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-8">My Courses</h1>
      
      <div v-if="enrolledCourses.length > 0" id="dashboard-enrolled-courses" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div 
          v-for="course in enrolledCourses" 
          :key="course.id"
          class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow cursor-pointer course-card"
          @click="openCourse(course)"
        >
          <div class="h-40 bg-gray-200">
            <img :src="course.image" :alt="course.title" class="w-full h-full object-cover">
          </div>
          <div class="p-6">
            <div class="flex items-center text-xs font-semibold tracking-wide uppercase text-blue-600 mb-1">
              {{ course.university }}
            </div>
            <h3 class="text-lg font-bold text-gray-900 mb-2">{{ course.title }}</h3>
            <div class="mt-4">
              <div class="w-full bg-gray-200 rounded-full h-2.5">
                <div class="bg-blue-600 h-2.5 rounded-full" style="width: 15%"></div>
              </div>
              <p class="text-xs text-gray-500 mt-2">15% Complete</p>
            </div>
            
            <div v-if="course.enrolled_as" class="mt-4 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium" :class="course.enrolled_as === 'audit' ? 'bg-gray-100 text-gray-800' : 'bg-green-100 text-green-800'">
              {{ course.enrolled_as === 'audit' ? 'Audit Mode' : 'Purchased' }}
            </div>
          </div>
        </div>
      </div>

      <div v-else class="text-center py-20 bg-white rounded-lg border border-dashed border-gray-300">
        <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
        <h3 class="mt-2 text-sm font-medium text-gray-900">No courses yet</h3>
        <p class="mt-1 text-sm text-gray-500">Get started by exploring our catalog.</p>
        <div class="mt-6">
          <button 
            @click="goHome"
            class="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-700 hover:bg-blue-800 focus:outline-none"
          >
            Explore Courses
          </button>
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
  name: 'LEARNER_DASHBOARD',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const enrolledCourses = computed(() => dataStore.enrolled_courses)

    async function openCourse(course) {
      store.selected_course_id = course.id
      store.setCurrentPageId('COURSE_HOME')
      await router.push({ name: 'COURSE_HOME', params: { id: course.id } })
    }

    async function goHome() {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      enrolledCourses,
      openCourse,
      goHome
    }
  }
}
</script>