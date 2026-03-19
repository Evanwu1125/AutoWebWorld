<template>
  <div class="min-h-screen bg-[#FDFBF7] flex flex-col items-center py-10 px-4">
    <div class="bg-white rounded-3xl shadow-xl w-full max-w-4xl overflow-hidden border border-gray-100 flex flex-col md:flex-row">
      
      <!-- Image Sidebar -->
      <div class="w-full md:w-1/3 h-64 md:h-auto bg-gray-200 relative">
        <img v-if="course" :src="course.image" class="w-full h-full object-cover" />
        <button id="course-back-list" @click="goBack" class="absolute top-4 left-4 bg-white/90 p-2 rounded-full shadow-md hover:bg-white transition-colors text-gray-600">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
        </button>
      </div>

      <!-- Content -->
      <div class="w-full md:w-2/3 p-8">
        <div class="flex items-center gap-3 mb-2">
          <span class="px-3 py-1 bg-orange-100 text-orange-600 rounded-full text-xs font-bold uppercase tracking-wider">
            {{ course?.level }}
          </span>
          <span class="text-gray-400 text-sm font-medium">{{ course?.total_sessions }} Sessions</span>
        </div>

        <h1 class="text-3xl font-bold text-gray-900 mb-4">{{ course?.title || 'Loading...' }}</h1>
        <p class="text-gray-500 mb-8 leading-relaxed">{{ course?.description }}</p>

        <!-- Progress Slider -->
        <div class="mb-8 p-6 bg-gray-50 rounded-2xl">
          <label class="block text-sm font-bold text-gray-700 mb-4 flex justify-between">
            <span>Progress Goal</span>
            <span class="text-orange-500">{{ progress }}%</span>
          </label>
          <input id="course-progress-slider" 
                 type="range" 
                 v-model.number="progress"
                 min="0" max="100" step="5"
                 @input="handleProgressChange"
                 class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-500" />
        </div>

        <!-- Goal Input -->
        <div class="mb-8">
          <label class="block text-sm font-bold text-gray-700 mb-2">What do you hope to learn?</label>
          <input id="course-goal-input" 
                 type="text" 
                 v-model="goal"
                 @input="handleGoalInput"
                 placeholder="e.g. Better stress management..."
                 class="w-full p-4 rounded-xl border border-gray-200 focus:border-orange-500 focus:ring-orange-500 bg-white" />
        </div>

        <!-- Actions -->
        <div class="flex flex-col sm:flex-row gap-4">
          <button id="course-enroll-button" 
                  @click="goToEnroll"
                  class="flex-1 bg-orange-500 hover:bg-orange-600 text-white font-bold py-4 px-6 rounded-xl shadow-lg hover:shadow-orange-500/30 transition-all">
            Enroll Now
          </button>
          
          <button id="course-reminder-button" 
                  @click="goToReminder"
                  class="flex-1 bg-white hover:bg-gray-50 text-gray-700 font-bold py-4 px-6 rounded-xl border border-gray-200 shadow-sm transition-colors flex items-center justify-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Set Reminder
          </button>
        </div>

      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'COURSE_DETAIL',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const courseId = computed(() => signatureStore.selected_course_id || route.params.id);
    const course = computed(() => dataStore.courses.find(c => c.id === courseId.value));

    const progress = ref(0);
    const goal = ref('');

    const handleProgressChange = () => {
      signatureStore.course_progress_percent = progress.value;
    };

    const handleGoalInput = () => {
      signatureStore.course_goal_text = goal.value;
    };

    const goToEnroll = async () => {
      await router.push({ name: 'COURSE_ENROLL_FORM' });
    };

    const goToReminder = async () => {
      await router.push({ name: 'REMINDER_FORM' });
    };

    const goBack = async () => {
      await router.push({ name: 'COURSES_LIST' });
    };

    onMounted(() => {
      if (courseId.value) {
        signatureStore.selected_course_id = courseId.value;
      }
    });

    return {
      course,
      progress,
      goal,
      handleProgressChange,
      handleGoalInput,
      goToEnroll,
      goToReminder,
      goBack
    };
  }
}
</script>