import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useDataStore = defineStore('data', () => {
  // Static Data
  const browse_sessions = ref([
    { id: 'session_1', title: 'Morning Meditation', description: 'Start your day with clarity and calm. A gentle practice to set positive intentions.', duration_min: 10, difficulty: 'Beginner', published_date: '2024-09-15', image: '/images/browse_sessions_session_1.jpg' },
    { id: 'session_2', title: 'Stress Relief', description: 'Release tension and find your center. Perfect for overwhelming moments.', duration_min: 15, difficulty: 'Beginner', published_date: '2024-09-22', image: '/images/browse_sessions_session_2.jpg' },
    { id: 'session_3', title: 'Anxiety Release', description: 'Let go of worry and embrace peace. Guided breathing to calm anxious thoughts.', duration_min: 20, difficulty: 'Intermediate', published_date: '2024-10-01', image: '/images/browse_sessions_session_3.jpg' },
    { id: 'session_4', title: 'Body Scan', description: 'Connect with your physical self. A journey through relaxation and awareness.', duration_min: 25, difficulty: 'Intermediate', published_date: '2024-10-08', image: '/images/browse_sessions_session_4.jpg' },
    { id: 'session_5', title: 'Loving Kindness', description: 'Cultivate compassion for yourself and others. Open your heart to warmth.', duration_min: 15, difficulty: 'Beginner', published_date: '2024-10-15', image: '/images/browse_sessions_session_5.jpg' },
    { id: 'session_6', title: 'Breath Awareness', description: 'Return to the basics. Simple yet powerful breathing meditation.', duration_min: 10, difficulty: 'Beginner', published_date: '2024-10-22', image: '/images/browse_sessions_session_6.jpg' },
    { id: 'session_7', title: 'Evening Wind Down', description: 'Transition from day to night. Release the day and prepare for rest.', duration_min: 15, difficulty: 'Beginner', published_date: '2024-10-29', image: '/images/browse_sessions_session_7.jpg' },
    { id: 'session_8', title: 'Mindful Walking', description: 'Meditation in motion. Bring awareness to each step and breath.', duration_min: 20, difficulty: 'Intermediate', published_date: '2024-11-05', image: '/images/browse_sessions_session_8.jpg' },
    { id: 'session_9', title: 'Gratitude Practice', description: 'Appreciate the good in your life. Shift perspective to abundance.', duration_min: 10, difficulty: 'Beginner', published_date: '2024-11-12', image: '/images/browse_sessions_session_9.jpg' },
    { id: 'session_10', title: 'Focus & Concentration', description: 'Sharpen your mental clarity. Train your mind to stay present.', duration_min: 15, difficulty: 'Intermediate', published_date: '2024-11-19', image: '/images/browse_sessions_session_10.jpg' },
    { id: 'session_11', title: 'Pain Management', description: 'Work with discomfort mindfully. Techniques to ease physical tension.', duration_min: 20, difficulty: 'Advanced', published_date: '2024-11-26', image: '/images/browse_sessions_session_11.jpg' },
    { id: 'session_12', title: 'Self-Compassion', description: 'Be kind to yourself. Develop a nurturing inner voice.', duration_min: 15, difficulty: 'Beginner', published_date: '2024-12-01', image: '/images/browse_sessions_session_12.jpg' },
    { id: 'session_13', title: 'Letting Go', description: 'Release what no longer serves you. Create space for new possibilities.', duration_min: 20, difficulty: 'Intermediate', published_date: '2024-12-05', image: '/images/browse_sessions_session_13.jpg' },
    { id: 'session_14', title: 'Deep Rest', description: 'Profound relaxation for body and mind. Restore your energy.', duration_min: 25, difficulty: 'Intermediate', published_date: '2024-12-08', image: '/images/browse_sessions_session_14.jpg' },
    { id: 'session_15', title: 'Creativity Boost', description: 'Unlock your creative potential. Clear mental blocks and inspire flow.', duration_min: 10, difficulty: 'Beginner', published_date: '2024-12-10', image: '/images/browse_sessions_session_15.jpg' },
    { id: 'session_16', title: 'Anger Release', description: 'Transform difficult emotions. Find calm in the storm.', duration_min: 15, difficulty: 'Intermediate', published_date: '2024-12-12', image: '/images/browse_sessions_session_16.jpg' },
    { id: 'session_17', title: 'Visualization Journey', description: 'Guided imagery for peace and healing. Travel to your inner sanctuary.', duration_min: 20, difficulty: 'Advanced', published_date: '2024-12-14', image: '/images/browse_sessions_session_17.jpg' },
    { id: 'session_18', title: 'Quick Reset', description: 'Fast relief when you need it most. Instant calm in 5 minutes.', duration_min: 5, difficulty: 'Beginner', published_date: '2024-12-16', image: '/images/browse_sessions_session_18.jpg' },
    { id: 'session_19', title: 'Mindful Eating', description: 'Transform your relationship with food. Savor each moment.', duration_min: 10, difficulty: 'Beginner', published_date: '2024-12-18', image: '/images/browse_sessions_session_19.jpg' },
    { id: 'session_20', title: 'Advanced Breathwork', description: 'Master pranayama techniques. Deep practice for experienced meditators.', duration_min: 30, difficulty: 'Advanced', published_date: '2024-12-20', image: '/images/browse_sessions_session_20.jpg' }
  ]);

  const courses = ref([
    { id: 'course_1', title: 'Basics of Meditation', description: 'Learn the fundamentals of meditation. Perfect for complete beginners starting their journey.', total_sessions: 10, level: 'Beginner', published_date: '2024-08-01', image: '/images/courses_course_1.jpg' },
    { id: 'course_2', title: 'Managing Anxiety', description: 'Practical tools to reduce worry and find calm. Evidence-based techniques for daily life.', total_sessions: 10, level: 'Beginner', published_date: '2024-08-15', image: '/images/courses_course_2.jpg' },
    { id: 'course_3', title: 'Sleep by Headspace', description: 'Improve your sleep quality naturally. Wind down techniques and bedtime meditations.', total_sessions: 7, level: 'Beginner', published_date: '2024-09-01', image: '/images/courses_course_3.jpg' },
    { id: 'course_4', title: 'Stress Management', description: 'Transform your relationship with stress. Build resilience and find balance.', total_sessions: 10, level: 'Beginner', published_date: '2024-09-10', image: '/images/courses_course_4.jpg' },
    { id: 'course_5', title: 'Focus & Productivity', description: 'Enhance concentration and mental clarity. Train your mind for peak performance.', total_sessions: 20, level: 'Intermediate', published_date: '2024-09-20', image: '/images/courses_course_5.jpg' },
    { id: 'course_6', title: 'Self-Esteem', description: 'Build confidence from within. Develop a healthier relationship with yourself.', total_sessions: 10, level: 'Beginner', published_date: '2024-10-01', image: '/images/courses_course_6.jpg' },
    { id: 'course_7', title: 'Mindful Eating', description: 'Change how you relate to food. Cultivate awareness and enjoyment in eating.', total_sessions: 8, level: 'Beginner', published_date: '2024-10-10', image: '/images/courses_course_7.jpg' },
    { id: 'course_8', title: 'Relationships', description: 'Improve connections with others. Communication and compassion practices.', total_sessions: 10, level: 'Intermediate', published_date: '2024-10-20', image: '/images/courses_course_8.jpg' },
    { id: 'course_9', title: 'Anger Management', description: 'Work skillfully with difficult emotions. Find calm in challenging moments.', total_sessions: 10, level: 'Intermediate', published_date: '2024-11-01', image: '/images/courses_course_9.jpg' },
    { id: 'course_10', title: 'Grief & Loss', description: 'Navigate difficult times with compassion. Gentle support for healing.', total_sessions: 6, level: 'Intermediate', published_date: '2024-11-08', image: '/images/courses_course_10.jpg' },
    { id: 'course_11', title: 'Creativity', description: 'Unlock your creative potential. Remove blocks and inspire innovation.', total_sessions: 10, level: 'Intermediate', published_date: '2024-11-15', image: '/images/courses_course_11.jpg' },
    { id: 'course_12', title: 'Acceptance', description: 'Make peace with what is. Find freedom through letting go.', total_sessions: 10, level: 'Intermediate', published_date: '2024-11-22', image: '/images/courses_course_12.jpg' },
    { id: 'course_13', title: 'Kindness', description: 'Cultivate compassion for all beings. Loving-kindness meditation practices.', total_sessions: 10, level: 'Beginner', published_date: '2024-11-28', image: '/images/courses_course_13.jpg' },
    { id: 'course_14', title: 'Patience', description: 'Develop calm in waiting. Transform frustration into peace.', total_sessions: 10, level: 'Intermediate', published_date: '2024-12-02', image: '/images/courses_course_14.jpg' },
    { id: 'course_15', title: 'Change', description: 'Navigate life transitions skillfully. Embrace uncertainty with confidence.', total_sessions: 10, level: 'Intermediate', published_date: '2024-12-05', image: '/images/courses_course_15.jpg' },
    { id: 'course_16', title: 'Appreciation', description: 'Discover gratitude in everyday moments. Shift to abundance mindset.', total_sessions: 10, level: 'Beginner', published_date: '2024-12-08', image: '/images/courses_course_16.jpg' },
    { id: 'course_17', title: 'Pain Management', description: 'Work mindfully with physical discomfort. Techniques for chronic pain.', total_sessions: 10, level: 'Advanced', published_date: '2024-12-11', image: '/images/courses_course_17.jpg' },
    { id: 'course_18', title: 'Pregnancy', description: 'Support for expecting mothers. Calm and connection during pregnancy.', total_sessions: 9, level: 'Beginner', published_date: '2024-12-14', image: '/images/courses_course_18.jpg' },
    { id: 'course_19', title: 'Sport & Performance', description: 'Mental training for athletes. Enhance focus and resilience.', total_sessions: 10, level: 'Intermediate', published_date: '2024-12-17', image: '/images/courses_course_19.jpg' },
    { id: 'course_20', title: 'Advanced Meditation', description: 'Deepen your practice. Techniques for experienced meditators.', total_sessions: 15, level: 'Advanced', published_date: '2024-12-20', image: '/images/courses_course_20.jpg' }
  ]);

  const sleep_tracks = ref([
    { id: 'sleep_1', title: 'Rainforest Dreams', description: 'Gentle rain and forest sounds to lull you into deep sleep.', duration_min: 45, type: 'Soundscape', intensity: 3, published_date: '2024-09-10', image: '/images/sleep_tracks_sleep_1.jpg' },
    { id: 'sleep_2', title: 'Ocean Waves', description: 'Rhythmic waves washing ashore. Natural white noise for peaceful rest.', duration_min: 60, type: 'Soundscape', intensity: 2, published_date: '2024-09-18', image: '/images/sleep_tracks_sleep_2.jpg' },
    { id: 'sleep_3', title: 'The Sleepy Dragon', description: 'A gentle bedtime story about a dragon who loves to nap in cozy caves.', duration_min: 35, type: 'Story', intensity: 1, published_date: '2024-09-25', image: '/images/sleep_tracks_sleep_3.jpg' },
    { id: 'sleep_4', title: 'Midnight Laundromat', description: 'The soothing hum of washing machines on a quiet night.', duration_min: 50, type: 'Soundscape', intensity: 4, published_date: '2024-10-02', image: '/images/sleep_tracks_sleep_4.jpg' },
    { id: 'sleep_5', title: 'Piano Lullabies', description: 'Soft piano melodies composed for restful sleep.', duration_min: 40, type: 'Music', intensity: 2, published_date: '2024-10-10', image: '/images/sleep_tracks_sleep_5.jpg' },
    { id: 'sleep_6', title: 'The Northern Lights', description: 'Journey to the Arctic in this calming bedtime tale.', duration_min: 30, type: 'Story', intensity: 1, published_date: '2024-10-18', image: '/images/sleep_tracks_sleep_6.jpg' },
    { id: 'sleep_7', title: 'Campfire Crackling', description: 'The warm, comforting sounds of a crackling campfire.', duration_min: 55, type: 'Soundscape', intensity: 3, published_date: '2024-10-25', image: '/images/sleep_tracks_sleep_7.jpg' },
    { id: 'sleep_8', title: 'Celestial Strings', description: 'Ethereal harp and string arrangements for deep relaxation.', duration_min: 45, type: 'Music', intensity: 2, published_date: '2024-11-01', image: '/images/sleep_tracks_sleep_8.jpg' },
    { id: 'sleep_9', title: 'The Lavender Fields', description: 'Wander through endless purple fields in this peaceful story.', duration_min: 28, type: 'Story', intensity: 1, published_date: '2024-11-08', image: '/images/sleep_tracks_sleep_9.jpg' },
    { id: 'sleep_10', title: 'Thunderstorm', description: 'Distant thunder and steady rain. Perfect for storm lovers.', duration_min: 60, type: 'Soundscape', intensity: 0, published_date: '2024-11-15', image: '/images/sleep_tracks_sleep_10.jpg' },
    { id: 'sleep_11', title: 'Cat Purring', description: 'The ultimate comfort sound. Continuous gentle purring.', duration_min: 50, type: 'Soundscape', intensity: 1, published_date: '2024-11-22', image: '/images/sleep_tracks_sleep_11.jpg' },
    { id: 'sleep_12', title: 'The Starlight Express', description: 'A magical train journey through the night sky.', duration_min: 32, type: 'Story', intensity: 2, published_date: '2024-11-28', image: '/images/sleep_tracks_sleep_12.jpg' },
    { id: 'sleep_13', title: 'Ambient Drones', description: 'Deep, resonant tones for profound relaxation.', duration_min: 60, type: 'Music', intensity: 4, published_date: '2024-12-02', image: '/images/sleep_tracks_sleep_13.jpg' },
    { id: 'sleep_14', title: 'Mountain Stream', description: 'Babbling brook flowing through peaceful mountain valleys.', duration_min: 55, type: 'Soundscape', intensity: 2, published_date: '2024-12-05', image: '/images/sleep_tracks_sleep_14.jpg' },
    { id: 'sleep_15', title: 'The Moonlit Garden', description: 'Explore a secret garden under the full moon.', duration_min: 30, type: 'Story', intensity: 1, published_date: '2024-12-08', image: '/images/sleep_tracks_sleep_15.jpg' },
    { id: 'sleep_16', title: 'Wind Chimes', description: 'Gentle breeze through bamboo chimes. Zen-like tranquility.', duration_min: 45, type: 'Soundscape', intensity: 2, published_date: '2024-12-11', image: '/images/sleep_tracks_sleep_16.jpg' },
    { id: 'sleep_17', title: 'Tibetan Bowls', description: 'Singing bowls and meditation bells for deep rest.', duration_min: 40, type: 'Music', intensity: 3, published_date: '2024-12-14', image: '/images/sleep_tracks_sleep_17.jpg' },
    { id: 'sleep_18', title: 'The Cloud Castle', description: 'Float away to a castle made of clouds in this dreamy tale.', duration_min: 35, type: 'Story', intensity: 1, published_date: '2024-12-16', image: '/images/sleep_tracks_sleep_18.jpg' },
    { id: 'sleep_19', title: 'Desert Night', description: 'Crickets and gentle wind in the peaceful desert evening.', duration_min: 50, type: 'Soundscape', intensity: 3, published_date: '2024-12-18', image: '/images/sleep_tracks_sleep_19.jpg' },
    { id: 'sleep_20', title: 'Weightless', description: 'Scientifically designed music to reduce anxiety and promote sleep.', duration_min: 45, type: 'Music', intensity: 2, published_date: '2024-12-20', image: '/images/sleep_tracks_sleep_20.jpg' }
  ]);

  const focus_sessions = ref([
    { id: 'focus_1', title: 'Deep Work', description: 'Intense focus music for complex tasks. Minimal distractions, maximum productivity.', duration_min: 90, has_music: true, music_type: 'Ambient', published_date: '2024-09-05', image: '/images/focus_sessions_focus_1.jpg' },
    { id: 'focus_2', title: 'Study Session', description: 'Lo-fi beats perfect for learning and retention.', duration_min: 60, has_music: true, music_type: 'Lo-Fi', published_date: '2024-09-12', image: '/images/focus_sessions_focus_2.jpg' },
    { id: 'focus_3', title: 'Creative Flow', description: 'Inspiring soundscapes to unlock your creativity.', duration_min: 45, has_music: true, music_type: 'Ambient', published_date: '2024-09-20', image: '/images/focus_sessions_focus_3.jpg' },
    { id: 'focus_4', title: 'Morning Productivity', description: 'Energizing piano melodies to start your day right.', duration_min: 30, has_music: true, music_type: 'Piano', published_date: '2024-09-28', image: '/images/focus_sessions_focus_4.jpg' },
    { id: 'focus_5', title: 'Coding Zone', description: 'Electronic ambient perfect for programming flow state.', duration_min: 120, has_music: true, music_type: 'Ambient', published_date: '2024-10-05', image: '/images/focus_sessions_focus_5.jpg' },
    { id: 'focus_6', title: 'Writing Time', description: 'Gentle background music that won\'t interrupt your thoughts.', duration_min: 60, has_music: true, music_type: 'Piano', published_date: '2024-10-12', image: '/images/focus_sessions_focus_6.jpg' },
    { id: 'focus_7', title: 'Quick Focus Boost', description: 'Short burst of concentration music for urgent tasks.', duration_min: 25, has_music: true, music_type: 'Lo-Fi', published_date: '2024-10-20', image: '/images/focus_sessions_focus_7.jpg' },
    { id: 'focus_8', title: 'Nature Sounds Focus', description: 'Forest ambience and birdsong for natural concentration.', duration_min: 60, has_music: true, music_type: 'Nature', published_date: '2024-10-28', image: '/images/focus_sessions_focus_8.jpg' },
    { id: 'focus_9', title: 'Exam Preparation', description: 'Calm, steady rhythms to help you study effectively.', duration_min: 90, has_music: true, music_type: 'Lo-Fi', published_date: '2024-11-05', image: '/images/focus_sessions_focus_9.jpg' },
    { id: 'focus_10', title: 'Afternoon Reset', description: 'Beat the post-lunch slump with uplifting focus music.', duration_min: 30, has_music: true, music_type: 'Piano', published_date: '2024-11-12', image: '/images/focus_sessions_focus_10.jpg' },
    { id: 'focus_11', title: 'Silent Focus', description: 'Guided focus session without music. Pure mindful work.', duration_min: 45, has_music: false, music_type: 'Ambient', published_date: '2024-11-18', image: '/images/focus_sessions_focus_11.jpg' },
    { id: 'focus_12', title: 'Reading Companion', description: 'Subtle soundscapes that enhance reading comprehension.', duration_min: 60, has_music: true, music_type: 'Ambient', published_date: '2024-11-24', image: '/images/focus_sessions_focus_12.jpg' },
    { id: 'focus_13', title: 'Design & Art', description: 'Inspiring music for visual creative work.', duration_min: 75, has_music: true, music_type: 'Ambient', published_date: '2024-11-30', image: '/images/focus_sessions_focus_13.jpg' },
    { id: 'focus_14', title: 'Pomodoro Timer', description: 'Classic 25-minute focus session with break reminders.', duration_min: 25, has_music: true, music_type: 'Lo-Fi', published_date: '2024-12-04', image: '/images/focus_sessions_focus_14.jpg' },
    { id: 'focus_15', title: 'Late Night Work', description: 'Mellow tones for burning the midnight oil productively.', duration_min: 90, has_music: true, music_type: 'Piano', published_date: '2024-12-08', image: '/images/focus_sessions_focus_15.jpg' },
    { id: 'focus_16', title: 'Brainstorming', description: 'Energetic yet non-intrusive music for idea generation.', duration_min: 30, has_music: true, music_type: 'Ambient', published_date: '2024-12-11', image: '/images/focus_sessions_focus_16.jpg' },
    { id: 'focus_17', title: 'Email & Admin', description: 'Light background music for routine tasks.', duration_min: 45, has_music: true, music_type: 'Lo-Fi', published_date: '2024-12-14', image: '/images/focus_sessions_focus_17.jpg' },
    { id: 'focus_18', title: 'Meeting Prep', description: 'Focused preparation music to organize your thoughts.', duration_min: 20, has_music: true, music_type: 'Piano', published_date: '2024-12-16', image: '/images/focus_sessions_focus_18.jpg' },
    { id: 'focus_19', title: 'Mindful Work', description: 'Combine meditation and productivity. Work with awareness.', duration_min: 60, has_music: false, music_type: 'Nature', published_date: '2024-12-18', image: '/images/focus_sessions_focus_19.jpg' },
    { id: 'focus_20', title: 'Power Hour', description: 'High-energy focus music for your most important hour.', duration_min: 60, has_music: true, music_type: 'Ambient', published_date: '2024-12-20', image: '/images/focus_sessions_focus_20.jpg' }
  ]);

  return {
    browse_sessions,
    courses,
    sleep_tracks,
    focus_sessions
  };
}, {
  persist: {
    storage: sessionStorage
  }
});