import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // --- MOCK DATA ---
  
  // 1. Courses (20 items)
  const courses = ref([
    {
      id: 'course_1',
      title: 'Machine Learning Specialization',
      instructor: 'Andrew Ng',
      university: 'Stanford University',
      rating: 4.9,
      students: 250000,
      level: 'beginner',
      duration: 30, // hours
      image: '/images/courses_course_1.jpg',
      description: 'Master Machine Learning fundamentals with this comprehensive course.',
      price: 49.99
    },
    {
      id: 'course_2',
      title: 'Python for Everybody',
      instructor: 'Dr. Charles Severance',
      university: 'University of Michigan',
      rating: 4.8,
      students: 1200000,
      level: 'beginner',
      duration: 20,
      image: '/images/courses_course_2.jpg',
      description: 'Learn to Program and Analyze Data with Python.',
      price: 39.99
    },
    {
      id: 'course_3',
      title: 'Google Data Analytics',
      instructor: 'Google Career Certificates',
      university: 'Google',
      rating: 4.8,
      students: 900000,
      level: 'beginner',
      duration: 180,
      image: '/images/courses_course_3.jpg',
      description: 'Get started in the high-growth field of data analytics.',
      price: 39.00 // subscription
    },
    {
      id: 'course_4',
      title: 'Deep Learning Specialization',
      instructor: 'Andrew Ng',
      university: 'DeepLearning.AI',
      rating: 4.9,
      students: 600000,
      level: 'intermediate',
      duration: 80,
      image: '/images/courses_course_4.jpg',
      description: 'Become a Deep Learning Expert. Master the fundamentals of deep learning.',
      price: 49.00
    },
    {
      id: 'course_5',
      title: 'Introduction to Psychology',
      instructor: 'Steve Joordens',
      university: 'University of Toronto',
      rating: 4.7,
      students: 300000,
      level: 'beginner',
      duration: 15,
      image: '/images/courses_course_5.jpg',
      description: 'This course highlights the most interesting experiments within the field of psychology.',
      price: 29.99
    },
    {
      id: 'course_6',
      title: 'The Science of Well-Being',
      instructor: 'Laurie Santos',
      university: 'Yale University',
      rating: 4.9,
      students: 4000000,
      level: 'beginner',
      duration: 19,
      image: '/images/courses_course_6.jpg',
      description: 'Engage in a series of challenges designed to increase your own happiness.',
      price: 0
    },
    {
      id: 'course_7',
      title: 'Financial Markets',
      instructor: 'Robert Shiller',
      university: 'Yale University',
      rating: 4.8,
      students: 800000,
      level: 'beginner',
      duration: 33,
      image: '/images/courses_course_7.jpg',
      description: 'An overview of the ideas, methods, and institutions that permit human society to manage risks.',
      price: 39.99
    },
    {
      id: 'course_8',
      title: 'Introduction to Marketing',
      instructor: 'Barbara Kahn',
      university: 'University of Pennsylvania',
      rating: 4.6,
      students: 200000,
      level: 'beginner',
      duration: 10,
      image: '/images/courses_course_8.jpg',
      description: 'Taught by three of Wharton\'s top faculty in the marketing department.',
      price: 49.99
    },
    {
      id: 'course_9',
      title: 'Algorithms Part I',
      instructor: 'Kevin Wayne',
      university: 'Princeton University',
      rating: 4.9,
      students: 700000,
      level: 'intermediate',
      duration: 54,
      image: '/images/courses_course_9.jpg',
      description: 'This course covers the essential information that every serious programmer needs.',
      price: 0
    },
    {
      id: 'course_10',
      title: 'Bitcoin and Cryptocurrency Technologies',
      instructor: 'Arvind Narayanan',
      university: 'Princeton University',
      rating: 4.7,
      students: 450000,
      level: 'intermediate',
      duration: 23,
      image: '/images/courses_course_10.jpg',
      description: 'To really understand what is special about Bitcoin, we need to understand how it works.',
      price: 29.99
    },
    {
      id: 'course_11',
      title: 'Graphic Design',
      instructor: 'David Underwood',
      university: 'University of Colorado Boulder',
      rating: 4.8,
      students: 150000,
      level: 'beginner',
      duration: 25,
      image: '/images/courses_course_11.jpg',
      description: 'Learn the fundamental skills required to make sophisticated graphic design.',
      price: 39.99
    },
    {
      id: 'course_12',
      title: 'Business Writing',
      instructor: 'Quentin McAndrew',
      university: 'University of Colorado Boulder',
      rating: 4.7,
      students: 100000,
      level: 'beginner',
      duration: 12,
      image: '/images/courses_course_12.jpg',
      description: 'Learn how to apply the top ten principles of good business writing.',
      price: 29.99
    },
    {
      id: 'course_13',
      title: 'Game Design and Development',
      instructor: 'Brian Winn',
      university: 'Michigan State University',
      rating: 4.8,
      students: 80000,
      level: 'beginner',
      duration: 60,
      image: '/images/courses_course_13.jpg',
      description: 'Learn theoretical and practical foundations of video game production.',
      price: 49.00
    },
    {
      id: 'course_14',
      title: 'Stanford Introduction to Food and Health',
      instructor: 'Maya Adam',
      university: 'Stanford University',
      rating: 4.7,
      students: 350000,
      level: 'beginner',
      duration: 6,
      image: '/images/courses_course_14.jpg',
      description: 'Explore innovative strategies for promoting healthy eating.',
      price: 19.99
    },
    {
      id: 'course_15',
      title: 'Learning How to Learn',
      instructor: 'Barbara Oakley',
      university: 'Deep Teaching Solutions',
      rating: 4.9,
      students: 2800000,
      level: 'beginner',
      duration: 15,
      image: '/images/courses_course_15.jpg',
      description: 'Powerful mental tools to help you master tough subjects.',
      price: 0
    },
    {
      id: 'course_16',
      title: 'Negotiation Skills',
      instructor: 'George Siedel',
      university: 'University of Michigan',
      rating: 4.8,
      students: 950000,
      level: 'beginner',
      duration: 16,
      image: '/images/courses_course_16.jpg',
      description: 'Learn to negotiate effectively for personal and professional success.',
      price: 39.99
    },
    {
      id: 'course_17',
      title: 'First Step Korean',
      instructor: 'Seung Hae Kang',
      university: 'Yonsei University',
      rating: 4.9,
      students: 1100000,
      level: 'beginner',
      duration: 12,
      image: '/images/courses_course_17.jpg',
      description: 'This is an elementary-level Korean language course.',
      price: 0
    },
    {
      id: 'course_18',
      title: 'English for Career Development',
      instructor: 'Brian McManus',
      university: 'University of Pennsylvania',
      rating: 4.8,
      students: 1300000,
      level: 'beginner',
      duration: 40,
      image: '/images/courses_course_18.jpg',
      description: 'Designed for non-native English speakers who are interested in advancing their careers.',
      price: 0
    },
    {
      id: 'course_19',
      title: 'Introduction to Philosophy',
      instructor: 'Duncan Pritchard',
      university: 'University of Edinburgh',
      rating: 4.6,
      students: 600000,
      level: 'beginner',
      duration: 20,
      image: '/images/courses_course_19.jpg',
      description: 'This course will introduce you to some of the main areas of research in contemporary philosophy.',
      price: 29.99
    },
    {
      id: 'course_20',
      title: 'Brand Management',
      instructor: 'Nader Tavassoli',
      university: 'London Business School',
      rating: 4.7,
      students: 250000,
      level: 'intermediate',
      duration: 18,
      image: '/images/courses_course_20.jpg',
      description: 'Learn how to build brands from a broad organizational perspective.',
      price: 39.99
    }
  ])

  // 2. Specializations (15 items)
  const specializations = ref([
    {
      id: 'spec_1',
      title: 'Data Science',
      university: 'Johns Hopkins University',
      rating: 4.5,
      courses_count: 10,
      duration: 11, // months
      image: '/images/specializations_spec_1.jpg',
      description: 'Launch Your Career in Data Science.',
      level: 'beginner'
    },
    {
      id: 'spec_2',
      title: 'Python 3 Programming',
      university: 'University of Michigan',
      rating: 4.7,
      courses_count: 5,
      duration: 5,
      image: '/images/specializations_spec_2.jpg',
      description: 'Become a Python 3 Programmer.',
      level: 'beginner'
    },
    {
      id: 'spec_3',
      title: 'Excel Skills for Business',
      university: 'Macquarie University',
      rating: 4.9,
      courses_count: 4,
      duration: 6,
      image: '/images/specializations_spec_3.jpg',
      description: 'Master Excel to add value to your company.',
      level: 'beginner'
    },
    {
      id: 'spec_4',
      title: 'Business Analytics',
      university: 'University of Pennsylvania',
      rating: 4.6,
      courses_count: 5,
      duration: 6,
      image: '/images/specializations_spec_4.jpg',
      description: 'Make data-driven business decisions.',
      level: 'beginner'
    },
    {
      id: 'spec_5',
      title: 'Digital Marketing',
      university: 'University of Illinois',
      rating: 4.6,
      courses_count: 7,
      duration: 8,
      image: '/images/specializations_spec_5.jpg',
      description: 'Master strategic marketing concepts and tools.',
      level: 'beginner'
    },
    {
      id: 'spec_6',
      title: 'Graphic Design Elements',
      university: 'CalArts',
      rating: 4.7,
      courses_count: 5,
      duration: 6,
      image: '/images/specializations_spec_6.jpg',
      description: 'Make Compelling Design.',
      level: 'beginner'
    },
    {
      id: 'spec_7',
      title: 'Applied Data Science with Python',
      university: 'University of Michigan',
      rating: 4.5,
      courses_count: 5,
      duration: 5,
      image: '/images/specializations_spec_7.jpg',
      description: 'Gain insight into your data.',
      level: 'intermediate'
    },
    {
      id: 'spec_8',
      title: 'Java Programming and SE',
      university: 'Duke University',
      rating: 4.6,
      courses_count: 5,
      duration: 5,
      image: '/images/specializations_spec_8.jpg',
      description: 'Take your first step towards a career in software development.',
      level: 'beginner'
    },
    {
      id: 'spec_9',
      title: 'Full Stack Web Development',
      university: 'Hong Kong University',
      rating: 4.7,
      courses_count: 6,
      duration: 7,
      image: '/images/specializations_spec_9.jpg',
      description: 'Build web and mobile applications.',
      level: 'intermediate'
    },
    {
      id: 'spec_10',
      title: 'Deep Learning',
      university: 'DeepLearning.AI',
      rating: 4.9,
      courses_count: 5,
      duration: 4,
      image: '/images/specializations_spec_10.jpg',
      description: 'Become a Deep Learning Expert.',
      level: 'intermediate'
    },
    {
      id: 'spec_11',
      title: 'Investment Management',
      university: 'University of Geneva',
      rating: 4.7,
      courses_count: 5,
      duration: 5,
      image: '/images/specializations_spec_11.jpg',
      description: 'Secure your future with wise investments.',
      level: 'beginner'
    },
    {
      id: 'spec_12',
      title: 'Project Management Principles',
      university: 'University of California, Irvine',
      rating: 4.8,
      courses_count: 4,
      duration: 6,
      image: '/images/specializations_spec_12.jpg',
      description: 'Launch your career in Project Management.',
      level: 'beginner'
    },
    {
      id: 'spec_13',
      title: 'Supply Chain Management',
      university: 'Rutgers University',
      rating: 4.6,
      courses_count: 5,
      duration: 6,
      image: '/images/specializations_spec_13.jpg',
      description: 'Master the fundamentals of Supply Chain Management.',
      level: 'beginner'
    },
    {
      id: 'spec_14',
      title: 'Human Resource Management',
      university: 'University of Minnesota',
      rating: 4.8,
      courses_count: 5,
      duration: 6,
      image: '/images/specializations_spec_14.jpg',
      description: 'Become a better manager of people.',
      level: 'beginner'
    },
    {
      id: 'spec_15',
      title: 'Creative Writing',
      university: 'Wesleyan University',
      rating: 4.7,
      courses_count: 5,
      duration: 6,
      image: '/images/specializations_spec_15.jpg',
      description: 'Craft your story with mastery.',
      level: 'beginner'
    }
  ])

  // 3. Professional Certificates (10 items)
  const professional_certs = ref([
    {
      id: 'cert_1',
      title: 'Google IT Support',
      provider: 'Google',
      rating: 4.8,
      duration: 6,
      image: '/images/professional_certs_cert_1.jpg',
      description: 'Prepare for a career in IT support.',
      price: 39
    },
    {
      id: 'cert_2',
      title: 'IBM Data Science',
      provider: 'IBM',
      rating: 4.6,
      duration: 10,
      image: '/images/professional_certs_cert_2.jpg',
      description: 'Kickstart your career in data science.',
      price: 39
    },
    {
      id: 'cert_3',
      title: 'Meta Front-End Developer',
      provider: 'Meta',
      rating: 4.7,
      duration: 7,
      image: '/images/professional_certs_cert_3.jpg',
      description: 'Launch a career as a front-end developer.',
      price: 49
    },
    {
      id: 'cert_4',
      title: 'Google UX Design',
      provider: 'Google',
      rating: 4.8,
      duration: 6,
      image: '/images/professional_certs_cert_4.jpg',
      description: 'Design user experiences for web and mobile.',
      price: 39
    },
    {
      id: 'cert_5',
      title: 'Salesforce Sales Operations',
      provider: 'Salesforce',
      rating: 4.6,
      duration: 4,
      image: '/images/professional_certs_cert_5.jpg',
      description: 'Build sales operations skills.',
      price: 49
    },
    {
      id: 'cert_6',
      title: 'Google Project Management',
      provider: 'Google',
      rating: 4.8,
      duration: 6,
      image: '/images/professional_certs_cert_6.jpg',
      description: 'Start a career in project management.',
      price: 39
    },
    {
      id: 'cert_7',
      title: 'Meta Back-End Developer',
      provider: 'Meta',
      rating: 4.7,
      duration: 8,
      image: '/images/professional_certs_cert_7.jpg',
      description: 'Get started with back-end development.',
      price: 49
    },
    {
      id: 'cert_8',
      title: 'IBM Cybersecurity Analyst',
      provider: 'IBM',
      rating: 4.6,
      duration: 8,
      image: '/images/professional_certs_cert_8.jpg',
      description: 'Launch your career in cybersecurity.',
      price: 39
    },
    {
      id: 'cert_9',
      title: 'Google Data Analytics',
      provider: 'Google',
      rating: 4.8,
      duration: 6,
      image: '/images/professional_certs_cert_9.jpg',
      description: 'Unlock insights from data.',
      price: 39
    },
    {
      id: 'cert_10',
      title: 'Intuit Bookkeeping',
      provider: 'Intuit',
      rating: 4.7,
      duration: 4,
      image: '/images/professional_certs_cert_10.jpg',
      description: 'Build a foundation in bookkeeping.',
      price: 29
    }
  ])

  // 4. Syllabi (Mock logic - get by course ID)
  const getSyllabus = (courseId) => {
    return [
      { week: 1, title: 'Introduction', duration: '2 hours' },
      { week: 2, title: 'Core Concepts', duration: '3 hours' },
      { week: 3, title: 'Advanced Topics', duration: '4 hours' },
      { week: 4, title: 'Final Project', duration: '5 hours' }
    ]
  }

  return {
    courses,
    specializations,
    professional_certs,
    getSyllabus
  }
}, {
  persist: {
    storage: sessionStorage
  }
})