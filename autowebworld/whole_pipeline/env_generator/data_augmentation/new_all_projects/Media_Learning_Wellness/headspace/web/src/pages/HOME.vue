<template>
  <div class="min-h-screen bg-[#FDFBF7] font-sans text-gray-800">
    <!-- Cookie Modal -->
    <CookieConsentModal 
      v-if="showCookieModal" 
      @accept="handleAcceptCookies"
      @decline="handleDeclineCookies" 
    />

    <!-- Header / Navigation -->
    <header class="bg-white/80 backdrop-blur-md sticky top-0 z-20 shadow-sm border-b border-gray-100">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
        <!-- Logo -->
        <div class="flex items-center gap-2">
          <div class="w-10 h-10 rounded-full bg-orange-500 flex items-center justify-center text-white font-bold text-2xl">H</div>
          <span class="text-xl font-bold tracking-tight text-gray-800">Headspace</span>
        </div>

        <!-- Navigation -->
        <nav class="hidden md:flex items-center gap-8">
          
          <!-- Direct Link -->
          <button id="nav-browse-direct" 
                  @click="goToBrowse"
                  class="text-gray-600 hover:text-orange-500 font-semibold transition-colors">
            Browse
          </button>

          <!-- Hover Menu (Explore) -->
          <div id="nav-explore-menu" class="relative group h-20 flex items-center">
            <button class="text-gray-600 group-hover:text-orange-500 font-semibold transition-colors flex items-center gap-1">
              Explore
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            <!-- Dropdown Content -->
            <div class="absolute top-full left-1/2 -translate-x-1/2 w-56 max-w-[calc(100vw-2rem)] bg-white rounded-xl shadow-xl border border-gray-100 p-2 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 transform translate-y-2 group-hover:translate-y-0">
              <div class="option-meditations p-3 rounded-lg hover:bg-orange-50 cursor-pointer flex items-center gap-3"
                   @click="handleExploreSelect('meditations')">
                <span class="text-xl">🧘</span>
                <span class="font-medium">Meditations</span>
              </div>
              <div class="option-courses p-3 rounded-lg hover:bg-orange-50 cursor-pointer flex items-center gap-3"
                   @click="handleExploreSelect('courses')">
                <span class="text-xl">📚</span>
                <span class="font-medium">Courses</span>
              </div>
              <div class="option-sleep p-3 rounded-lg hover:bg-orange-50 cursor-pointer flex items-center gap-3"
                   @click="handleExploreSelect('sleep')">
                <span class="text-xl">😴</span>
                <span class="font-medium">Sleep</span>
              </div>
              <div class="option-focus p-3 rounded-lg hover:bg-orange-50 cursor-pointer flex items-center gap-3"
                   @click="handleExploreSelect('focus')">
                <span class="text-xl">🎧</span>
                <span class="font-medium">Focus</span>
              </div>
            </div>
          </div>

          <!-- Click Dropdown (More) -->
          <div class="relative">
            <button id="nav-more-dropdown" 
                    @click="toggleMoreMenu"
                    class="text-gray-600 hover:text-orange-500 font-semibold transition-colors flex items-center gap-1">
              More
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 transition-transform duration-200" :class="{'rotate-180': isMoreMenuOpen}" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            <!-- Dropdown Content -->
            <div v-if="isMoreMenuOpen" 
                 class="absolute top-full right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-gray-100 p-2 z-50 animate-fade-in-down">
              <div id="nav-more-sleep" 
                   @click="handleMoreSelect('sleep')"
                   class="p-3 rounded-lg hover:bg-blue-50 cursor-pointer flex items-center gap-3">
                <span class="text-xl">🌙</span>
                <span class="font-medium">Sleep</span>
              </div>
              <div id="nav-more-focus" 
                   @click="handleMoreSelect('focus')"
                   class="p-3 rounded-lg hover:bg-blue-50 cursor-pointer flex items-center gap-3">
                <span class="text-xl">🎯</span>
                <span class="font-medium">Focus</span>
              </div>
              <div id="nav-more-profile" 
                   @click="handleMoreSelect('profile')"
                   class="p-3 rounded-lg hover:bg-blue-50 cursor-pointer flex items-center gap-3">
                <span class="text-xl">👤</span>
                <span class="font-medium">Profile</span>
              </div>
            </div>
          </div>
        </nav>
      </div>
    </header>

    <!-- Main Content -->
    <main>
      <!-- Hero Section -->
      <section class="relative h-[600px] flex items-center overflow-hidden">
        <!-- Background Image with Overlay -->
        <div class="absolute inset-0 z-0">
          <img src="/images/Lake.jpg" alt="Peaceful lake meditation scene" class="w-full h-full object-cover" />
          <div class="absolute inset-0 bg-gradient-to-r from-orange-50/90 to-transparent"></div>
        </div>

        <div class="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 w-full">
          <div class="max-w-xl">
            <h1 class="text-5xl md:text-6xl font-bold text-gray-900 mb-6 leading-tight">
              Find your <span class="text-orange-500">peace</span> of mind.
            </h1>
            <p class="text-xl text-gray-700 mb-8 leading-relaxed">
              Meditation and mindfulness for any mind, any mood, any goal. 
              Start your journey to a happier, healthier life today.
            </p>
            <div class="flex gap-4">
              <button @click="goToBrowse" 
                      class="bg-orange-500 hover:bg-orange-600 text-white font-bold py-4 px-8 rounded-full shadow-lg hover:shadow-orange-500/30 transition-all duration-300 transform hover:-translate-y-1">
                Start Meditating
              </button>
              <button class="bg-white hover:bg-gray-50 text-gray-800 font-bold py-4 px-8 rounded-full shadow-md hover:shadow-lg transition-all duration-300 border border-gray-100">
                Learn More
              </button>
            </div>
          </div>
        </div>
      </section>

      <!-- Features Grid -->
      <section class="py-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <h2 class="text-3xl font-bold text-center mb-12 text-gray-800">Mindfulness for every moment</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
          <!-- Card 1 -->
          <div class="bg-white p-8 rounded-3xl shadow-sm hover:shadow-xl transition-shadow duration-300 border border-gray-50 text-center">
            <div class="w-16 h-16 bg-yellow-100 rounded-full flex items-center justify-center text-3xl mx-auto mb-6">☀️</div>
            <h3 class="text-xl font-bold mb-3">Wake Up</h3>
            <p class="text-gray-600">Start your day with intention and clarity through our morning guided sessions.</p>
          </div>
          <!-- Card 2 -->
          <div class="bg-white p-8 rounded-3xl shadow-sm hover:shadow-xl transition-shadow duration-300 border border-gray-50 text-center">
             <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center text-3xl mx-auto mb-6">🌙</div>
            <h3 class="text-xl font-bold mb-3">Sleep Soundly</h3>
            <p class="text-gray-600">Drift off with sleepcasts, music, and wind-down exercises designed for rest.</p>
          </div>
          <!-- Card 3 -->
          <div class="bg-white p-8 rounded-3xl shadow-sm hover:shadow-xl transition-shadow duration-300 border border-gray-50 text-center">
             <div class="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center text-3xl mx-auto mb-6">🧠</div>
            <h3 class="text-xl font-bold mb-3">Find Focus</h3>
            <p class="text-gray-600">Get in the zone with focus music and exercises to boost your productivity.</p>
          </div>
        </div>
      </section>
    </main>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import CookieConsentModal from '../components/CookieConsentModal.vue';

export default {
  name: 'HOME',
  components: {
    CookieConsentModal
  },
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    // State
    const isMoreMenuOpen = ref(false);

    // Computed
    const showCookieModal = computed(() => signatureStore.cookie_consent_given === null);

    // Actions
    const handleAcceptCookies = () => {
      // Effect: set cookie_consent_given = true
      signatureStore.cookie_consent_given = true;
      // Action: ACT_HOME_ACCEPT_COOKIES
      console.log('Cookies accepted');
    };

    const handleDeclineCookies = () => {
      // Not strictly in FSM, but good UX
      signatureStore.cookie_consent_given = false;
    };

    const goToBrowse = async () => {
      if (!signatureStore.cookie_consent_given) return;
      // Action: ACT_HOME_GO_BROWSE_DIRECT
      await router.push({ name: 'BROWSE' });
    };

    const handleExploreSelect = async (value) => {
      if (!signatureStore.cookie_consent_given) return;
      // Action: ACT_HOME_GO_COURSES_HOVER
      if (value === 'courses' || value === 'meditations') {
        await router.push({ name: 'COURSES_LIST' });
      } else if (value === 'sleep') {
        // Fallback if user selects sleep here, though FSM maps this specific action to COURSES_LIST primarily
        // But let's follow FSM strictly: ACT_HOME_GO_COURSES_HOVER -> to: COURSES_LIST
        // The FSM implies this hover menu navigates to COURSES_LIST regardless of option selected in this specific action context?
        // Checking FSM: yes, to: "COURSES_LIST". 
        await router.push({ name: 'COURSES_LIST' });
      } else if (value === 'focus') {
        await router.push({ name: 'COURSES_LIST' });
      }
    };

    const toggleMoreMenu = () => {
      isMoreMenuOpen.value = !isMoreMenuOpen.value;
    };

    const handleMoreSelect = async (value) => {
      if (!signatureStore.cookie_consent_given) return;
      
      if (value === 'sleep') {
        // Action: ACT_HOME_GO_SLEEP_MENU
        await router.push({ name: 'SLEEP_LIST' });
      } else if (value === 'focus') {
        // Action: ACT_HOME_GO_FOCUS_MENU
        await router.push({ name: 'FOCUS_LIST' });
      }
      isMoreMenuOpen.value = false;
    };

    return {
      showCookieModal,
      isMoreMenuOpen,
      handleAcceptCookies,
      handleDeclineCookies,
      goToBrowse,
      handleExploreSelect,
      toggleMoreMenu,
      handleMoreSelect
    };
  }
}
</script>

<style scoped>
.animate-fade-in-down {
  animation: fadeInDown 0.2s ease-out;
}
@keyframes fadeInDown {
  from { opacity: 0; transform: translateY(-10px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>