# Development Checklist

## 1. Project Setup
- [ ] Install dependencies: `pnpm install pinia-plugin-persistedstate`
- [ ] Configure `src/main.js` (Pinia, PersistedState, Router)
- [ ] Setup `src/App.vue` (Layout, Router View, Interceptors)
- [ ] Setup `src/app.css` (Tailwind directives)

## 2. Store Implementation
- [ ] `src/stores/signature.js`: FSM state, current page, interceptor flags.
- [ ] `src/stores/data.js`: Mock data for Accounts, Beneficiaries, Transactions, Cards, Exchange Pairs, Topup Methods.

## 3. FSM Runtime
- [ ] `src/fsm/FSMRuntime.js`: Runtime engine for preconditions and effects validation (simplified for Vue).

## 4. Page Implementation
- [ ] `src/pages/HOME.vue`: Cookie consent, Dashboard navigation.
- [ ] `src/pages/ACCOUNTS_DASHBOARD.vue`: Accounts list, filters (checkbox, slider, sort), search.
- [ ] `src/pages/ACCOUNT_DETAIL.vue`: Single account view, actions.
- [ ] `src/pages/PAYMENTS_LIST.vue`: Beneficiaries list, filters, search.
- [ ] `src/pages/BENEFICIARY_DETAIL.vue`: Single beneficiary view.
- [ ] `src/pages/TRANSFER_FORM.vue`: Transfer input form.
- [ ] `src/pages/TRANSFER_REVIEW.vue`: Review transfer.
- [ ] `src/pages/TRANSFER_SUCCESS.vue`: Success state.
- [ ] `src/pages/EXCHANGE_DASHBOARD.vue`: Exchange pairs list, filters.
- [ ] `src/pages/EXCHANGE_FORM.vue`: Exchange input form.
- [ ] `src/pages/EXCHANGE_REVIEW.vue`: Review exchange.
- [ ] `src/pages/EXCHANGE_SUCCESS.vue`: Success state.
- [ ] `src/pages/CARDS_LIST.vue`: Cards list, filters.
- [ ] `src/pages/CARD_DETAIL.vue`: Single card view, actions (Freeze, Limits).
- [ ] `src/pages/CARD_FREEZE_FORM.vue`: Freeze reason.
- [ ] `src/pages/CARD_FREEZE_SUCCESS.vue`: Success state.
- [ ] `src/pages/CARD_LIMITS_FORM.vue`: ATM/POS limits sliders.
- [ ] `src/pages/CARD_LIMITS_SUCCESS.vue`: Success state.
- [ ] `src/pages/TOPUP_METHOD_LIST.vue`: Topup methods list, filters.
- [ ] `src/pages/TOPUP_FORM.vue`: Topup amount.
- [ ] `src/pages/TOPUP_REVIEW.vue`: Review topup.
- [ ] `src/pages/TOPUP_SUCCESS.vue`: Success state.

## 5. Components
- [ ] `src/components/PermissionModal.vue`: For location permission.
- [ ] `src/components/CookieConsentModal.vue`: For cookie consent.
- [ ] `src/components/NavBar.vue`: Global navigation (if needed, or per page).

## 6. Routing
- [ ] `src/router/index.js`: Define all routes matching FSM page IDs.

## 7. Validation
- [ ] Verify selectors match `fsm.json`.
- [ ] Verify actions update store correctly.
- [ ] Verify navigation guards.
- [ ] Lint and Build.