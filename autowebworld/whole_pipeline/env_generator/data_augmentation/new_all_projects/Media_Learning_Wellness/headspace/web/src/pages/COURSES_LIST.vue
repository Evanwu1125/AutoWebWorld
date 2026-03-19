<template>
  <div class="min-h-screen bg-[#FDFBF7] pb-20">
    <!-- Header -->
    <header class="bg-white sticky top-0 z-20 shadow-sm">
      <div class="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <button id="courses-back-home" @click="goHome" class="text-gray-600 hover:text-orange-500 font-medium flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clip-rule="evenodd" />
          </svg>
          Back Home
        </button>
        <h1 class="text-xl font-bold text-gray-800">Courses</h1>
        <div class="w-20"></div>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 pt-8">
      <!-- Search & Sort -->
      <div class="flex flex-col md:flex-row gap-4 mb-8">
        <div class="relative flex-1">
          <input id="courses-search-input" 
                 type="text" 
                 v-model="searchQuery"
                 @keydown.enter="handleSearch"
                 placeholder="Search courses..." 
                 class="w-full pl-10 pr-4 py-3 rounded-xl border-gray-200 focus:border-orange-500 focus:ring-orange-500 transition-all shadow-sm" />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 absolute left-3 top-3.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        <div class="relative">
          <button id="courses-sort-dropdown" 
                  @click="toggleSortMenu"
                  class="bg-white px-6 py-3 rounded-xl border border-gray-200 shadow-sm flex items-center gap-2 hover:border-orange-500 transition-colors w-full md:w-auto justify-between">
            <span class="text-gray-700 font-medium">Sort by: {{ currentSortLabel }}</span>
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>
          
          <div v-if="isSortMenuOpen" class="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-gray-100 z-30 py-2">
            <div id="courses-sort-popular" @click="handleSort('popular')" class="px-4 py-2 hover:bg-orange-50 cursor-pointer">Popular</div>
            <div id="courses-sort-new" @click="handleSort('new')" class="px-4 py-2 hover:bg-orange-50 cursor-pointer">Newest</div>
            <div id="courses-sort-total_sessions-inc" @click="handleSort('total_sessions')" class="px-4 py-2 hover:bg-orange-50 cursor-pointer">Total sessions</div>
          </div>
        </div>
      </div>

      <!-- Filters -->
      <div class="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 mb-8">
        <h3 class="font-bold text-gray-800 mb-4">Filters</h3>
        <div class="flex flex-col md:flex-row gap-8">
          <!-- Checkbox Filter -->
          <div class="flex-1">
            <label class="block text-sm font-semibold text-gray-500 mb-3 uppercase tracking-wide">Level</label>
            <div class="flex gap-4">
              <label class="flex items-center gap-2 cursor-pointer group">
                <input id="courses-filter-beginner-checkbox" type="checkbox" 
                       v-model="filters.beginner"
                       class="w-5 h-5 rounded text-orange-500 focus:ring-orange-500 border-gray-300 transition-all" />
                <span class="group-hover:text-orange-600">Beginner Friendly</span>
              </label>
            </div>
          </div>

          <!-- Slider Filter -->
          <div class="flex-1">
            <label class="block text-sm font-semibold text-gray-500 mb-3 uppercase tracking-wide">
              Min Sessions: {{ filters.length }}
            </label>
            <input id="courses-length-slider" type="range" 
                   v-model.number="filters.length"
                   min="0" max="20" step="1"
                   class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-500" />
            <div class="flex justify-between text-xs text-gray-400 mt-2">
              <span>0</span>
              <span>20+</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Courses List -->
      <div id="courses-list" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div v-for="course in filteredCourses" :key="course.id"
             :class="getRowClass(course)"
             class="bg-white rounded-2xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-300 group cursor-pointer border border-gray-50"
             @click="openCourse(course)">
          
          <!-- Image -->
          <div class="h-48 overflow-hidden relative">
            <img :src="course.image" :alt="course.title" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
            <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent p-4">
               <span class="text-white font-bold text-sm">{{ course.total_sessions }} Sessions</span>
            </div>
          </div>

          <!-- Content -->
          <div class="p-5">
            <div class="flex items-center gap-2 mb-2">
              <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-orange-50 text-orange-600">
                {{ course.level }}
              </span>
              <span class="px-2 py-0.5 rounded text-[10px] font-medium bg-gray-100 text-gray-600">
                {{ course.published_date }}
              </span>
            </div>
            <h3 class="font-bold text-lg text-gray-900 mb-2 group-hover:text-orange-500 transition-colors">
              {{ course.title }}
            </h3>
            <p class="text-gray-500 text-sm line-clamp-2">{{ course.description }}</p>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="filteredCourses.length === 0" class="text-center py-20">
        <div class="text-6xl mb-4">📚</div>
        <h3 class="text-xl font-bold text-gray-800 mb-2">No courses found</h3>
        <p class="text-gray-500">Try adjusting your filters.</p>
      </div>

    </main>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'COURSES_LIST',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const searchQuery = ref('');
    const isSortMenuOpen = ref(false);
    const currentSort = ref(null);
    const filters = ref({
      beginner: false,
      length: 0
    });

    watch(filters, () => {
      signatureStore.courses_filters_applied = true;
    }, { deep: true });

    watch(searchQuery, (newVal) => {
      if(newVal) signatureStore.courses_has_searched = true;
    });

    const filteredCourses = computed(() => {
      let result = dataStore.courses;

      if (searchQuery.value) {
        const query = searchQuery.value.toLowerCase();
        result = result.filter(c => c.title.toLowerCase().includes(query));
        if (result.length > 0) signatureStore.matched_course_id = result[0].id;
      }

      if (filters.value.beginner) {
        result = result.filter(c => c.level === 'Beginner');
      }

      if (filters.value.length > 0) {
        result = result.filter(c => c.total_sessions >= filters.value.length);
      }

      if (currentSort.value === 'new') {
        result = [...result].sort((a, b) => new Date(b.published_date) - new Date(a.published_date));
      } else if (currentSort.value === 'total_sessions') {
        result = [...result].sort((a, b) => a.total_sessions - b.total_sessions);
      } else if (currentSort.value === 'popular') {
        result = [...result].sort((a, b) => a.id.localeCompare(b.id));
      }

      return result;
    });

    const currentSortLabel = computed(() => {
      if (!currentSort.value) return 'Default';
      if (currentSort.value === 'new') return 'Newest';
      if (currentSort.value === 'total_sessions') return 'Shortest';
      return 'Popular';
    });

    const toggleSortMenu = () => isSortMenuOpen.value = !isSortMenuOpen.value;

    const handleSort = (type) => {
      currentSort.value = type;
      signatureStore.courses_filters_applied = true;
      isSortMenuOpen.value = false;
    };

    const handleSearch = () => {
      signatureStore.courses_has_searched = true;
      if (filteredCourses.value.length > 0) {
        signatureStore.matched_course_id = filteredCourses.value[0].id;
      }
    };

    const getRowClass = (course) => {
      const classes = [`data-id-${course.id}`];
      if (signatureStore.courses_filters_applied) classes.push('course-row-filtered');
      if (signatureStore.courses_has_searched && course.id === signatureStore.matched_course_id) classes.push('course-row-matched');
      if (!signatureStore.courses_filters_applied && !signatureStore.courses_has_searched) classes.push('course-row-visible');
      return classes.join(' ');
    };

    const openCourse = async (course) => {
      signatureStore.courses_filters_applied = null;
      signatureStore.courses_has_searched = null;
      signatureStore.courses_viewport_anchor_id = null;
      signatureStore.selected_course_id = course.id;
      await router.push({ name: 'COURSE_DETAIL', params: { id: course.id } });
    };

    const goHome = async () => {
      await router.push({ name: 'HOME' });
    };

    return {
      searchQuery,
      filters,
      isSortMenuOpen,
      currentSortLabel,
      filteredCourses,
      toggleSortMenu,
      handleSort,
      handleSearch,
      getRowClass,
      openCourse,
      goHome
    };
  }
}
</script>