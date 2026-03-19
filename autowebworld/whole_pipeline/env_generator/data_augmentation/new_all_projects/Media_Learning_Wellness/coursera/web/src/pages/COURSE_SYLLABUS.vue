<template>
  <div class="min-h-screen bg-gray-50 pb-12">
    <!-- Breadcrumbs / Back Tab -->
    <div class="bg-white border-b border-gray-200">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <nav class="-mb-px flex space-x-8">
          <div 
            id="tab-overview"
            @click="goBack"
            class="border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm cursor-pointer flex items-center"
          >
            <svg class="h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Overview
          </div>
          <div class="border-blue-500 text-blue-600 whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm">
            Syllabus
          </div>
        </nav>
      </div>
    </div>

    <div v-if="course" class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 mt-8">
      <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ course.title }}</h1>
      <p class="text-xl text-gray-500 mb-8">Syllabus - What you will learn</p>
      
      <div class="space-y-6">
        <div 
          v-for="week in syllabus" 
          :key="week.week"
          class="bg-white rounded-lg shadow-sm border border-gray-200 p-6 transition hover:shadow-md"
        >
          <div class="flex items-center justify-between mb-4">
            <span class="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-semibold uppercase tracking-wide">
              Week {{ week.week }}
            </span>
            <span class="text-sm text-gray-500 flex items-center">
              <svg class="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {{ week.duration }}
            </span>
          </div>
          <h3 class="text-xl font-bold text-gray-900 mb-2">{{ week.title }}</h3>
          <p class="text-gray-600">
            In this module, you will learn about {{ week.title.toLowerCase() }} through video lectures, readings, and quizzes.
          </p>
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
  name: 'COURSE_SYLLABUS',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const courseId = route.params.id || store.selected_course_id
    const course = computed(() => dataStore.courses.find(c => c.id === courseId))
    const syllabus = computed(() => dataStore.getSyllabus(courseId))

    async function goBack() {
      store.setCurrentPageId('COURSE_DETAIL')
      await router.push({ name: 'COURSE_DETAIL', params: { id: courseId } })
    }

    return {
      course,
      syllabus,
      goBack
    }
  }
}
</script>