<template>
  <div class="min-h-screen bg-gray-50 pb-12">
    <!-- Nav -->
    <nav class="bg-white shadow-sm">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <div id="header-logo-home" class="flex-shrink-0 flex items-center cursor-pointer" @click="goHome">
              <span class="text-2xl font-bold text-blue-700">Coursera</span>
            </div>
            <div class="ml-10 flex items-baseline space-x-4">
               <span v-if="course" class="text-lg font-medium text-gray-900">{{ course.title }}</span>
            </div>
          </div>
        </div>
      </div>
    </nav>

    <div v-if="course" class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
      <!-- Course Content -->
      <div class="flex flex-col lg:flex-row gap-8">
        <!-- Sidebar Menu -->
        <div class="lg:w-64 flex-shrink-0">
          <div class="bg-white shadow rounded-lg overflow-hidden">
            <div class="p-4 bg-gray-50 border-b border-gray-200 font-bold text-gray-700">
              Course Material
            </div>
            <ul class="divide-y divide-gray-200">
              <li class="p-4 hover:bg-gray-50 cursor-pointer text-blue-700 font-medium border-l-4 border-blue-700 bg-blue-50">
                Week 1
              </li>
              <li class="p-4 hover:bg-gray-50 cursor-pointer text-gray-600">
                Week 2
              </li>
              <li class="p-4 hover:bg-gray-50 cursor-pointer text-gray-600">
                Week 3
              </li>
              <li class="p-4 hover:bg-gray-50 cursor-pointer text-gray-600">
                Week 4
              </li>
            </ul>
          </div>

          <div class="mt-6 bg-white shadow rounded-lg overflow-hidden">
             <div 
               id="course-home-rate-this-course"
               @click="goToRating"
               class="p-4 hover:bg-gray-50 cursor-pointer text-blue-700 font-medium flex items-center"
             >
               <span class="text-yellow-400 mr-2">★</span> Rate this course
             </div>
          </div>
        </div>

        <!-- Main Content Area -->
        <div class="flex-1">
          <div class="bg-white shadow rounded-lg p-8">
            <h2 class="text-2xl font-bold text-gray-900 mb-6">Welcome to Week 1</h2>
            
            <div class="prose max-w-none text-gray-600">
              <p class="mb-4">
                Welcome to the course! In this first week, we will cover the foundational concepts necessary for success.
                Please start by watching the introductory video below.
              </p>

              <div class="aspect-w-16 aspect-h-9 bg-gray-200 rounded-lg flex items-center justify-center mb-6 h-64">
                <span class="text-gray-500">Video Player Placeholder</span>
              </div>

              <h3 class="text-lg font-bold text-gray-900 mb-2">To-Do List</h3>
              <ul class="list-disc pl-5 space-y-2">
                <li>Watch: Introduction to {{ course.title }} (10 min)</li>
                <li>Read: Course Syllabus (15 min)</li>
                <li>Quiz: Week 1 Assessment (30 min)</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'COURSE_HOME',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const courseId = route.params.id || store.selected_course_id
    const course = computed(() => dataStore.courses.find(c => c.id === courseId))

    async function goToRating() {
      store.setCurrentPageId('COURSE_RATING_FORM')
      await router.push({ name: 'COURSE_RATING_FORM', params: { id: courseId } })
    }

    async function goHome() {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      course,
      goToRating,
      goHome
    }
  }
}
</script>