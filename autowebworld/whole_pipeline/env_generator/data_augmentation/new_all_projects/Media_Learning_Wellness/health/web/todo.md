# Development Checklist

## 1. Project Setup
- [ ] Install `pinia-plugin-persistedstate`
- [ ] Configure `src/main.js` with pinia persistence
- [ ] Create `src/stores/signature.js` (FSM State)
- [ ] Create `src/stores/data.js` (Mock Data - Large dataset)
- [ ] Configure `src/router/index.js` with all 30 routes and guards
- [ ] Create `src/components/widgets/DateTimePicker.vue` (Verify existence or import usage) - *Template has it? Instruction says "DateTimePicker.vue is ALREADY in the template"* -> I will verify later.

## 2. Pages (30 Total)
- [ ] HOME (Interceptors: Cookie)
- [ ] LOGIN (Interceptors: Location)
- [ ] DASHBOARD
- [ ] VISIT_TYPE_SELECTION (Interceptors: Location)
- [ ] PROVIDER_LIST (Search, Scroll, Filter, Sort)
- [ ] PROVIDER_DETAIL
- [ ] SCHEDULE_APPOINTMENT (Date/Time Picker)
- [ ] SCHEDULE_REVIEW
- [ ] SCHEDULE_VISIT_SUCCESS
- [ ] INSTANT_VISIT_TRIAGE
- [ ] INSTANT_VISIT_QUEUE
- [ ] INSTANT_VISIT_SUCCESS
- [ ] PRESCRIPTION_LIST (Search, Scroll, Filter)
- [ ] PRESCRIPTION_DETAIL
- [ ] PRESCRIPTION_RENEWAL_REVIEW
- [ ] PRESCRIPTION_RENEWAL_SUCCESS
- [ ] MENTAL_HEALTH_LIST (Search, Scroll, Filter)
- [ ] MENTAL_HEALTH_DETAIL
- [ ] MENTAL_HEALTH_SCHEDULE
- [ ] MENTAL_HEALTH_REVIEW
- [ ] MENTAL_HEALTH_BOOKING_SUCCESS
- [ ] APPOINTMENTS_LIST (Search, Scroll, Filter)
- [ ] APPOINTMENT_DETAIL
- [ ] BILLING_OVERVIEW (Search, Scroll)
- [ ] BILL_DETAIL
- [ ] BILL_PAYMENT
- [ ] BILL_PAYMENT_SUCCESS
- [ ] BENEFITS_OVERVIEW (Filter)
- [ ] SETTINGS_ACCOUNT
- [ ] SETTINGS_INSURANCE

## 3. Global Components
- [ ] Navigation Bar (if not part of layout)
- [ ] Footer
- [ ] Permission Modal (Location)
- [ ] Cookie Consent Modal

## 4. Validation
- [ ] Verify all FSM actions are mapped
- [ ] Verify DOM selectors match `gui_procedure`
- [ ] Lint and Build