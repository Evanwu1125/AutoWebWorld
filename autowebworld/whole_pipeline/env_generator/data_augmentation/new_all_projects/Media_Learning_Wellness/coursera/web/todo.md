# Development Checklist

## Phase 1: Setup
- [ ] Install `pinia-plugin-persistedstate` (Done)
- [ ] Configure `src/main.js` with Pinia persistence
- [ ] Create `src/stores/signature.js` (FSM State Store)
- [ ] Create `src/stores/data.js` (Mock Data Store with 15-20 items per collection)
- [ ] Create `src/router/index.js` (Route definitions for all 23 pages)

## Phase 2: Page Implementation (Batch 1 - Core & Discovery)
- [ ] `src/pages/HOME.vue` (Cookie Consent, Hero, Nav)
- [ ] `src/pages/LOGIN.vue`
- [ ] `src/pages/COURSE_DISCOVERY.vue` (Filters, Search, List, Location Permission)
- [ ] `src/pages/COURSE_DETAIL.vue` (Course info, Enrollment selection)
- [ ] `src/pages/COURSE_SYLLABUS.vue`

## Phase 3: Enrollment & Payment Flow
- [ ] `src/pages/ENROLLMENT_OPTIONS.vue`
- [ ] `src/pages/PAYMENT_DETAILS.vue`
- [ ] `src/pages/ORDER_REVIEW.vue`
- [ ] `src/pages/ENROLL_COURSE_SUCCESS.vue`
- [ ] `src/pages/AUDIT_CONFIRM.vue`
- [ ] `src/pages/AUDIT_COURSE_SUCCESS.vue`

## Phase 4: Course Learning & Rating
- [ ] `src/pages/COURSE_HOME.vue`
- [ ] `src/pages/COURSE_RATING_FORM.vue`
- [ ] `src/pages/COURSE_RATING_SUBMITTED_SUCCESS.vue`
- [ ] `src/pages/LEARNER_DASHBOARD.vue`

## Phase 5: Specialization & Professional Cert Flows
- [ ] `src/pages/SPECIALIZATION_LIST.vue`
- [ ] `src/pages/SPECIALIZATION_DETAIL.vue`
- [ ] `src/pages/SPECIALIZATION_SUBSCRIBE_PAYMENT.vue`
- [ ] `src/pages/SPECIALIZATION_SUBSCRIBE_SUCCESS.vue`
- [ ] `src/pages/PROFESSIONAL_CERT_LIST.vue`
- [ ] `src/pages/PROFESSIONAL_CERT_DETAIL.vue`
- [ ] `src/pages/PROFESSIONAL_CERT_ENROLL_PAYMENT.vue`
- [ ] `src/pages/ENROLL_PROFESSIONAL_CERT_SUCCESS.vue`

## Phase 6: Validation
- [ ] Verify all FSM actions are mapped
- [ ] Verify all GUI selectors exist
- [ ] Verify data store population
- [ ] Build & Lint