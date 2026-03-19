# Development Checklist

## Phase 1: Setup
- [x] Install dependencies (pinia-plugin-persistedstate)
- [x] Create style.txt
- [ ] Configure main.js with persistence plugin
- [ ] Create router/index.js with all 19 routes
- [ ] Create stores/signature.js (FSM State)
- [ ] Create stores/data.js (Mock Data with image paths)

## Phase 2: Page Implementation (19 Pages)
1.  **HOME**: Landing page, cookie consent, navigation.
2.  **TICKETS_LIST**: Table of tickets, filters, sorting, search, location permission.
3.  **TICKET_DETAIL**: Reply, status/priority update, navigate to sub-flows.
4.  **NEW_TICKET_FORM**: Input subject, description, priority, group.
5.  **NEW_TICKET_REVIEW**: Review and submit new ticket.
6.  **TICKET_CREATION_SUCCESS**: Success message.
7.  **REPLY_REVIEW**: Review reply content.
8.  **REPLY_SENT_SUCCESS**: Success message.
9.  **ASSIGN_TICKET**: Select agent.
10. **ASSIGN_SUCCESS**: Success message.
11. **MERGE_TICKET_SELECT**: Search and select ticket to merge with.
12. **MERGE_TICKET_CONFIRM**: Confirm merge.
13. **MERGE_SUCCESS**: Success message.
14. **CONTACTS_LIST**: Table of contacts, filters, search.
15. **CONTACT_DETAIL**: Contact info placeholder.
16. **NEW_CONTACT_FORM**: Input name, email, segment.
17. **NEW_CONTACT_REVIEW**: Review and submit new contact.
18. **CONTACT_CREATION_SUCCESS**: Success message.
19. **DASHBOARD**: Charts/Widgets simulation, location permission.

## Phase 3: Components & Widgets
- [ ] DateTimePicker (Verify existence)
- [ ] Permission Modal (Location)
- [ ] Cookie Consent Modal

## Phase 4: Mock Data Generation
- [ ] Generate 20+ Tickets
- [ ] Generate 15+ Contacts
- [ ] Generate Agents
- [ ] Ensure ImageGetter paths for all entities

## Phase 5: Verification
- [ ] Check all GUI selectors
- [ ] Verify FSM logic mapping
- [ ] Build & Lint