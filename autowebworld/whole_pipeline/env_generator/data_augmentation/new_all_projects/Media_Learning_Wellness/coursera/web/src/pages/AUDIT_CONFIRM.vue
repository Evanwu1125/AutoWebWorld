<template>
  <div class="min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
    <div class="sm:mx-auto sm:w-full sm:max-w-md">
      <h2 class="mt-6 text-center text-3xl font-extrabold text-gray-900">
        Audit Course
      </h2>
      <p class="mt-2 text-center text-sm text-gray-600">
        You are about to audit <span class="font-bold">{{ course?.title }}</span>
      </p>
    </div>

    <div class="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
      <div class="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
        <div class="rounded-md bg-blue-50 p-4 mb-6">
          <div class="flex">
            <div class="flex-shrink-0">
              <svg class="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="ml-3 flex-1 md:flex md:justify-between">
              <p class="text-sm text-blue-700">
                Auditing allows you to view all course materials for free. You will not receive a certificate or be able to submit graded assignments.
              </p>
            </div>
          </div>
        </div>

        <div class="space-y-6">
          <div class="flex items-start">
            <div class="flex items-center h-5">
              <input 
                id="audit-terms-checkbox" 
                name="terms" 
                type="checkbox" 
                class="focus:ring-blue-500 h-4 w-4 text-blue-600 border-gray-300 rounded"
                @change="handleTermsCheck"
              >
            </div>
            <div class="ml-3 text-sm">
              <label for="audit-terms-checkbox" class="font-medium text-gray-700">I understand the audit terms</label>
              <p class="text-gray-500">I agree to the Honor Code and Terms of Service.</p>
            </div>
          </div>

          <div>
            <button 
              id="audit-confirm-button"
              type="button" 
              class="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-700 hover:bg-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              :disabled="!store.audit_terms_confirmed"
              @click="confirmAudit"
            >
              Start Auditing
            </button>
          </div>

          <div class="text-center">
            <div 
              id="back-to-course-link"
              class="text-sm font-medium text-blue-600 hover:text-blue-500 cursor-pointer"
              @click="cancel"
            >
              Cancel
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
  name: 'AUDIT_CONFIRM',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const courseId = route.params.id || store.selected_course_id
    const course = computed(() => dataStore.courses.find(c => c.id === courseId))

    function handleTermsCheck(e) {
      store.audit_terms_confirmed = e.target.checked
    }

    async function confirmAudit() {
      if (store.audit_terms_confirmed) {
        // Add to enrolled list in mock
        dataStore.enrolled_courses.push({
          ...course.value,
          enrolled_as: 'audit'
        })
        
        store.setCurrentPageId('AUDIT_COURSE_SUCCESS')
        await router.push({ name: 'AUDIT_COURSE_SUCCESS' })
      }
    }

    async function cancel() {
      store.setCurrentPageId('COURSE_DETAIL')
      await router.push({ name: 'COURSE_DETAIL', params: { id: courseId } })
    }

    return {
      store,
      course,
      handleTermsCheck,
      confirmAudit,
      cancel
    }
  }
}
</script>