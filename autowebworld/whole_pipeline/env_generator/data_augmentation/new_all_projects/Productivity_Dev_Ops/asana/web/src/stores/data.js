import { defineStore } from 'pinia'
import { ref } from 'vue'

// Composition API format - users data
const users = ref([
    { id: 'u1', name: 'Me', avatar: '/images/User.jpg' },
    { id: 'u2', name: 'Alice Smith', avatar: '/images/user-2.jpg' },
    { id: 'u3', name: 'Bob Johnson', avatar: '/images/User.jpg' },
    { id: 'u4', name: 'Charlie Brown', avatar: '/images/User.jpg' },
    { id: 'u5', name: 'Diana Prince', avatar: '/images/DianaPrince.jpg' },
    { id: 'u6', name: 'Evan Wright', avatar: '/images/User.jpg' },
    { id: 'u7', name: 'Fiona Gallagher', avatar: '/images/FionaGallagher.jpg' },
    { id: 'u8', name: 'George Lucas', avatar: '/images/GeorgeLucas.jpg' },
])

// Composition API format - projects data
const projects = ref([
    { id: 'p1', name: 'Project 1: Mobile App', description: 'Description for project 1. This involves multiple stakeholders.', status: 'Active', priority: 95, due_date: '2025-01-01T00:00:00.000Z', owner_id: 'u1', image: '/images/projects_p1.jpg' },
    { id: 'p2', name: 'Project 2: Website Redesign', description: 'Description for project 2. This involves multiple stakeholders.', status: 'Active', priority: 88, due_date: '2025-02-01T00:00:00.000Z', owner_id: 'u2', image: '/images/projects_p2.jpg' },
    { id: 'p3', name: 'Project 3: Mobile App', description: 'Description for project 3. This involves multiple stakeholders.', status: 'Active', priority: 92, due_date: '2025-03-01T00:00:00.000Z', owner_id: 'u3', image: '/images/projects_p3.jpg' },
    { id: 'p4', name: 'Project 4: Marketing Campaign', description: 'Description for project 4. This involves multiple stakeholders.', status: 'Active', priority: 78, due_date: '2025-04-01T00:00:00.000Z', owner_id: 'u4', image: '/images/projects_p4.jpg' },
    { id: 'p5', name: 'Project 5: Q3 Roadmap', description: 'Description for project 5. This involves multiple stakeholders.', status: 'Active', priority: 85, due_date: '2025-05-01T00:00:00.000Z', owner_id: 'u5', image: '/images/projects_p5.jpg' },
    { id: 'p6', name: 'Project 6: Audit', description: 'Description for project 6. This involves multiple stakeholders.', status: 'Active', priority: 72, due_date: '2025-06-01T00:00:00.000Z', owner_id: 'u6', image: '/images/projects_p6.jpg' },
    { id: 'p7', name: 'Project 7: Hiring Plan', description: 'Description for project 7. This involves multiple stakeholders.', status: 'Active', priority: 90, due_date: '2025-07-01T00:00:00.000Z', owner_id: 'u7', image: '/images/projects_p7.jpg' },
    { id: 'p8', name: 'Project 8: Website Redesign', description: 'Description for project 8. This involves multiple stakeholders.', status: 'Active', priority: 82, due_date: '2025-08-01T00:00:00.000Z', owner_id: 'u8', image: '/images/projects_p8.jpg' },
    { id: 'p9', name: 'Project 9: Mobile App', description: 'Description for project 9. This involves multiple stakeholders.', status: 'Active', priority: 87, due_date: '2025-09-01T00:00:00.000Z', owner_id: 'u1', image: '/images/projects_p9.jpg' },
    { id: 'p10', name: 'Project 10: Marketing Campaign', description: 'Description for project 10. This involves multiple stakeholders.', status: 'Active', priority: 75, due_date: '2025-10-01T00:00:00.000Z', owner_id: 'u2', image: '/images/projects_p10.jpg' },
    { id: 'p11', name: 'Project 11: Q3 Roadmap', description: 'Description for project 11. This involves multiple stakeholders.', status: 'Active', priority: 93, due_date: '2025-11-01T00:00:00.000Z', owner_id: 'u3', image: '/images/projects_p11.jpg' },
    { id: 'p12', name: 'Project 12: Audit', description: 'Description for project 12. This involves multiple stakeholders.', status: 'Active', priority: 68, due_date: '2025-12-01T00:00:00.000Z', owner_id: 'u4', image: '/images/projects_p12.jpg' },
    { id: 'p13', name: 'Project 13: Hiring Plan', description: 'Description for project 13. This involves multiple stakeholders.', status: 'Active', priority: 80, due_date: '2025-01-15T00:00:00.000Z', owner_id: 'u5', image: '/images/projects_p13.jpg' },
    { id: 'p14', name: 'Project 14: Website Redesign', description: 'Description for project 14. This involves multiple stakeholders.', status: 'Active', priority: 86, due_date: '2025-02-15T00:00:00.000Z', owner_id: 'u6', image: '/images/projects_p14.jpg' },
    { id: 'p15', name: 'Project 15: Mobile App', description: 'Description for project 15. This involves multiple stakeholders.', status: 'Active', priority: 91, due_date: '2025-03-15T00:00:00.000Z', owner_id: 'u7', image: '/images/projects_p15.jpg' },
    { id: 'p16', name: 'Project 16: Marketing Campaign', description: 'Description for project 16. This involves multiple stakeholders.', status: 'Active', priority: 76, due_date: '2025-04-15T00:00:00.000Z', owner_id: 'u8', image: '/images/projects_p16.jpg' },
    { id: 'p17', name: 'Project 17: Q3 Roadmap', description: 'Description for project 17. This involves multiple stakeholders.', status: 'Active', priority: 84, due_date: '2025-05-15T00:00:00.000Z', owner_id: 'u1', image: '/images/projects_p17.jpg' },
    { id: 'p18', name: 'Project 18: Audit', description: 'Description for project 18. This involves multiple stakeholders.', status: 'Active', priority: 70, due_date: '2025-06-15T00:00:00.000Z', owner_id: 'u2', image: '/images/projects_p18.jpg' },
    { id: 'p19', name: 'Project 19: Hiring Plan', description: 'Description for project 19. This involves multiple stakeholders.', status: 'Active', priority: 89, due_date: '2025-07-15T00:00:00.000Z', owner_id: 'u3', image: '/images/projects_p19.jpg' },
    { id: 'p20', name: 'Project 20: Website Redesign', description: 'Description for project 20. This involves multiple stakeholders.', status: 'Active', priority: 81, due_date: '2025-08-15T00:00:00.000Z', owner_id: 'u4', image: '/images/projects_p20.jpg' },
])

// Composition API format - sections data
const sections = ref([
    { id: 'p1-s1', name: 'To Do', project_id: 'p1' },
    { id: 'p1-s2', name: 'In Progress', project_id: 'p1' },
    { id: 'p1-s3', name: 'Done', project_id: 'p1' },
    { id: 'p2-s1', name: 'To Do', project_id: 'p2' },
    { id: 'p2-s2', name: 'In Progress', project_id: 'p2' },
    { id: 'p2-s3', name: 'Done', project_id: 'p2' },
    { id: 'p3-s1', name: 'To Do', project_id: 'p3' },
    { id: 'p3-s2', name: 'In Progress', project_id: 'p3' },
    { id: 'p3-s3', name: 'Done', project_id: 'p3' },
    { id: 'p4-s1', name: 'To Do', project_id: 'p4' },
    { id: 'p4-s2', name: 'In Progress', project_id: 'p4' },
    { id: 'p4-s3', name: 'Done', project_id: 'p4' },
    { id: 'p5-s1', name: 'To Do', project_id: 'p5' },
    { id: 'p5-s2', name: 'In Progress', project_id: 'p5' },
    { id: 'p5-s3', name: 'Done', project_id: 'p5' },
    { id: 'p6-s1', name: 'To Do', project_id: 'p6' },
    { id: 'p6-s2', name: 'In Progress', project_id: 'p6' },
    { id: 'p6-s3', name: 'Done', project_id: 'p6' },
    { id: 'p7-s1', name: 'To Do', project_id: 'p7' },
    { id: 'p7-s2', name: 'In Progress', project_id: 'p7' },
    { id: 'p7-s3', name: 'Done', project_id: 'p7' },
    { id: 'p8-s1', name: 'To Do', project_id: 'p8' },
    { id: 'p8-s2', name: 'In Progress', project_id: 'p8' },
    { id: 'p8-s3', name: 'Done', project_id: 'p8' },
    { id: 'p9-s1', name: 'To Do', project_id: 'p9' },
    { id: 'p9-s2', name: 'In Progress', project_id: 'p9' },
    { id: 'p9-s3', name: 'Done', project_id: 'p9' },
    { id: 'p10-s1', name: 'To Do', project_id: 'p10' },
    { id: 'p10-s2', name: 'In Progress', project_id: 'p10' },
    { id: 'p10-s3', name: 'Done', project_id: 'p10' },
    { id: 'p11-s1', name: 'To Do', project_id: 'p11' },
    { id: 'p11-s2', name: 'In Progress', project_id: 'p11' },
    { id: 'p11-s3', name: 'Done', project_id: 'p11' },
    { id: 'p12-s1', name: 'To Do', project_id: 'p12' },
    { id: 'p12-s2', name: 'In Progress', project_id: 'p12' },
    { id: 'p12-s3', name: 'Done', project_id: 'p12' },
    { id: 'p13-s1', name: 'To Do', project_id: 'p13' },
    { id: 'p13-s2', name: 'In Progress', project_id: 'p13' },
    { id: 'p13-s3', name: 'Done', project_id: 'p13' },
    { id: 'p14-s1', name: 'To Do', project_id: 'p14' },
    { id: 'p14-s2', name: 'In Progress', project_id: 'p14' },
    { id: 'p14-s3', name: 'Done', project_id: 'p14' },
    { id: 'p15-s1', name: 'To Do', project_id: 'p15' },
    { id: 'p15-s2', name: 'In Progress', project_id: 'p15' },
    { id: 'p15-s3', name: 'Done', project_id: 'p15' },
    { id: 'p16-s1', name: 'To Do', project_id: 'p16' },
    { id: 'p16-s2', name: 'In Progress', project_id: 'p16' },
    { id: 'p16-s3', name: 'Done', project_id: 'p16' },
    { id: 'p17-s1', name: 'To Do', project_id: 'p17' },
    { id: 'p17-s2', name: 'In Progress', project_id: 'p17' },
    { id: 'p17-s3', name: 'Done', project_id: 'p17' },
    { id: 'p18-s1', name: 'To Do', project_id: 'p18' },
    { id: 'p18-s2', name: 'In Progress', project_id: 'p18' },
    { id: 'p18-s3', name: 'Done', project_id: 'p18' },
    { id: 'p19-s1', name: 'To Do', project_id: 'p19' },
    { id: 'p19-s2', name: 'In Progress', project_id: 'p19' },
    { id: 'p19-s3', name: 'Done', project_id: 'p19' },
    { id: 'p20-s1', name: 'To Do', project_id: 'p20' },
    { id: 'p20-s2', name: 'In Progress', project_id: 'p20' },
    { id: 'p20-s3', name: 'Done', project_id: 'p20' },
])

// Composition API format - tasks data
const tasks = ref([
    { id: 't1', name: 'Task 1: Fix Bug', description: 'Detailed description for task 1. Needs attention.', project_id: 'p1', section_id: 'p1-s1', assignee_id: 'u1', priority: 98, due_date: '2026-01-07T00:00:00.000Z', completed: false, image: '/images/tasks_t1.jpg' },
    { id: 't2', name: 'Task 2: Write Docs', description: 'Detailed description for task 2. Needs attention.', project_id: 'p1', section_id: 'p1-s2', assignee_id: 'u1', priority: 85, due_date: '2026-01-07T00:00:00.000Z', completed: true, image: '/images/tasks_t2.jpg' },
    { id: 't3', name: 'Task 3: Design Icon', description: 'Detailed description for task 3. Needs attention.', project_id: 'p2', section_id: 'p2-s1', assignee_id: 'u1', priority: 91, due_date: '2026-01-07T00:00:00.000Z', completed: false, image: '/images/tasks_t3.jpg' },
    { id: 't4', name: 'Task 4: Meeting', description: 'Detailed description for task 4. Needs attention.', project_id: 'p2', section_id: 'p2-s2', assignee_id: 'u1', priority: 73, due_date: '2026-01-07T00:00:00.000Z', completed: false, image: '/images/tasks_t4.jpg' },
    { id: 't5', name: 'Task 5: Email Client', description: 'Detailed description for task 5. Needs attention.', project_id: 'p3', section_id: 'p3-s1', assignee_id: 'u1', priority: 87, due_date: '2026-01-07T00:00:00.000Z', completed: false, image: '/images/tasks_t5.jpg' },
    { id: 't6', name: 'Task 6: Deploy Server', description: 'Detailed description for task 6. Needs attention.', project_id: 'p3', section_id: 'p3-s2', assignee_id: 'u1', priority: 79, due_date: '2025-01-15T00:00:00.000Z', completed: true, image: '/images/tasks_t6.jpg' },
    { id: 't7', name: 'Task 7: Fix Bug', description: 'Detailed description for task 7. Needs attention.', project_id: 'p4', section_id: 'p4-s1', assignee_id: 'u1', priority: 94, due_date: '2025-01-22T00:00:00.000Z', completed: false, image: '/images/tasks_t7.jpg' },
    { id: 't8', name: 'Task 8: Write Docs', description: 'Detailed description for task 8. Needs attention.', project_id: 'p4', section_id: 'p4-s2', assignee_id: 'u1', priority: 82, due_date: '2025-01-29T00:00:00.000Z', completed: false, image: '/images/tasks_t8.jpg' },
    { id: 't9', name: 'Task 9: Design Icon', description: 'Detailed description for task 9. Needs attention.', project_id: 'p5', section_id: 'p5-s1', assignee_id: 'u1', priority: 89, due_date: '2025-02-05T00:00:00.000Z', completed: false, image: '/images/tasks_t9.jpg' },
    { id: 't10', name: 'Task 10: Meeting', description: 'Detailed description for task 10. Needs attention.', project_id: 'p5', section_id: 'p5-s2', assignee_id: 'u1', priority: 71, due_date: '2025-02-12T00:00:00.000Z', completed: true, image: '/images/tasks_t10.jpg' },
    { id: 't11', name: 'Task 11: Email Client', description: 'Detailed description for task 11. Needs attention.', project_id: 'p6', section_id: 'p6-s1', assignee_id: 'u1', priority: 96, due_date: '2025-02-19T00:00:00.000Z', completed: false, image: '/images/tasks_t11.jpg' },
    { id: 't12', name: 'Task 12: Deploy Server', description: 'Detailed description for task 12. Needs attention.', project_id: 'p6', section_id: 'p6-s2', assignee_id: 'u1', priority: 77, due_date: '2025-02-26T00:00:00.000Z', completed: false, image: '/images/tasks_t12.jpg' },
    { id: 't13', name: 'Task 13: Fix Bug', description: 'Detailed description for task 13. Needs attention.', project_id: 'p7', section_id: 'p7-s1', assignee_id: 'u1', priority: 83, due_date: '2025-03-05T00:00:00.000Z', completed: false, image: '/images/tasks_t13.jpg' },
    { id: 't14', name: 'Task 14: Write Docs', description: 'Detailed description for task 14. Needs attention.', project_id: 'p7', section_id: 'p7-s2', assignee_id: 'u1', priority: 90, due_date: '2025-03-12T00:00:00.000Z', completed: false, image: '/images/tasks_t14.jpg' },
    { id: 't15', name: 'Task 15: Design Icon', description: 'Detailed description for task 15. Needs attention.', project_id: 'p8', section_id: 'p8-s1', assignee_id: 'u1', priority: 74, due_date: '2025-03-19T00:00:00.000Z', completed: false, image: '/images/tasks_t15.jpg' },
    { id: 't16', name: 'Task 16: Meeting', description: 'Detailed description for task 16. Needs attention.', project_id: 'p8', section_id: 'p8-s2', assignee_id: 'u1', priority: 88, due_date: '2025-03-26T00:00:00.000Z', completed: true, image: '/images/tasks_t16.jpg' },
    { id: 't17', name: 'Task 17: Email Client', description: 'Detailed description for task 17. Needs attention.', project_id: 'p9', section_id: 'p9-s1', assignee_id: 'u1', priority: 80, due_date: '2025-04-02T00:00:00.000Z', completed: false, image: '/images/tasks_t17.jpg' },
    { id: 't18', name: 'Task 18: Deploy Server', description: 'Detailed description for task 18. Needs attention.', project_id: 'p9', section_id: 'p9-s2', assignee_id: 'u1', priority: 92, due_date: '2025-04-09T00:00:00.000Z', completed: false, image: '/images/tasks_t18.jpg' },
    { id: 't19', name: 'Task 19: Fix Bug', description: 'Detailed description for task 19. Needs attention.', project_id: 'p10', section_id: 'p10-s1', assignee_id: 'u1', priority: 69, due_date: '2025-04-16T00:00:00.000Z', completed: false, image: '/images/tasks_t19.jpg' },
    { id: 't20', name: 'Task 20: Write Docs', description: 'Detailed description for task 20. Needs attention.', project_id: 'p10', section_id: 'p10-s2', assignee_id: 'u1', priority: 86, due_date: '2025-04-23T00:00:00.000Z', completed: true, image: '/images/tasks_t20.jpg' },
    { id: 't21', name: 'Task 21: Design Icon', description: 'Detailed description for task 21. Needs attention.', project_id: 'p11', section_id: 'p11-s1', assignee_id: 'u1', priority: 97, due_date: '2025-04-30T00:00:00.000Z', completed: false, image: '/images/tasks_t21.jpg' },
    { id: 't22', name: 'Task 22: Meeting', description: 'Detailed description for task 22. Needs attention.', project_id: 'p11', section_id: 'p11-s2', assignee_id: 'u1', priority: 75, due_date: '2025-05-07T00:00:00.000Z', completed: false, image: '/images/tasks_t22.jpg' },
    { id: 't23', name: 'Task 23: Email Client', description: 'Detailed description for task 23. Needs attention.', project_id: 'p12', section_id: 'p12-s1', assignee_id: 'u1', priority: 84, due_date: '2025-05-14T00:00:00.000Z', completed: false, image: '/images/tasks_t23.jpg' },
    { id: 't24', name: 'Task 24: Deploy Server', description: 'Detailed description for task 24. Needs attention.', project_id: 'p12', section_id: 'p12-s2', assignee_id: 'u1', priority: 81, due_date: '2025-05-21T00:00:00.000Z', completed: false, image: '/images/tasks_t24.jpg' },
    { id: 't25', name: 'Task 25: Fix Bug', description: 'Detailed description for task 25. Needs attention.', project_id: 'p13', section_id: 'p13-s1', assignee_id: 'u1', priority: 93, due_date: '2025-05-28T00:00:00.000Z', completed: false, image: '/images/tasks_t25.jpg' },
    { id: 't26', name: 'Task 26: Write Docs', description: 'Detailed description for task 26. Needs attention.', project_id: 'p13', section_id: 'p13-s2', assignee_id: 'u1', priority: 72, due_date: '2025-06-04T00:00:00.000Z', completed: true, image: '/images/tasks_t26.jpg' },
    { id: 't27', name: 'Task 27: Design Icon', description: 'Detailed description for task 27. Needs attention.', project_id: 'p14', section_id: 'p14-s1', assignee_id: 'u1', priority: 78, due_date: '2025-06-11T00:00:00.000Z', completed: false, image: '/images/tasks_t27.jpg' },
    { id: 't28', name: 'Task 28: Meeting', description: 'Detailed description for task 28. Needs attention.', project_id: 'p14', section_id: 'p14-s2', assignee_id: 'u1', priority: 95, due_date: '2025-06-18T00:00:00.000Z', completed: false, image: '/images/tasks_t28.jpg' },
    { id: 't29', name: 'Task 29: Email Client', description: 'Detailed description for task 29. Needs attention.', project_id: 'p15', section_id: 'p15-s1', assignee_id: 'u1', priority: 70, due_date: '2025-06-25T00:00:00.000Z', completed: false, image: '/images/tasks_t29.jpg' },
    { id: 't30', name: 'Task 30: Deploy Server', description: 'Detailed description for task 30. Needs attention.', project_id: 'p15', section_id: 'p15-s2', assignee_id: 'u1', priority: 99, due_date: '2025-07-02T00:00:00.000Z', completed: true, image: '/images/tasks_t30.jpg' },
    { id: 't31', name: 'Task 31: Fix Bug', description: 'Detailed description for task 31. Needs attention.', project_id: 'p16', section_id: 'p16-s1', assignee_id: 'u1', priority: 76, due_date: '2025-07-09T00:00:00.000Z', completed: false, image: '/images/tasks_t31.jpg' },
    { id: 't32', name: 'Task 32: Write Docs', description: 'Detailed description for task 32. Needs attention.', project_id: 'p16', section_id: 'p16-s2', assignee_id: 'u1', priority: 100, due_date: '2025-07-16T00:00:00.000Z', completed: false, image: '/images/tasks_t32.jpg' },
    { id: 't33', name: 'Task 33: Design Icon', description: 'Detailed description for task 33. Needs attention.', project_id: 'p17', section_id: 'p17-s1', assignee_id: 'u1', priority: 68, due_date: '2025-07-23T00:00:00.000Z', completed: false, image: '/images/tasks_t33.jpg' },
    { id: 't34', name: 'Task 34: Meeting', description: 'Detailed description for task 34. Needs attention.', project_id: 'p17', section_id: 'p17-s2', assignee_id: 'u1', priority: 67, due_date: '2025-07-30T00:00:00.000Z', completed: false, image: '/images/tasks_t34.jpg' },
    { id: 't35', name: 'Task 35: Email Client', description: 'Detailed description for task 35. Needs attention.', project_id: 'p18', section_id: 'p18-s1', assignee_id: 'u1', priority: 66, due_date: '2025-08-06T00:00:00.000Z', completed: false, image: '/images/tasks_t35.jpg' },
    { id: 't36', name: 'Task 36: Deploy Server', description: 'Detailed description for task 36. Needs attention.', project_id: 'p18', section_id: 'p18-s2', assignee_id: 'u1', priority: 65, due_date: '2025-08-13T00:00:00.000Z', completed: true, image: '/images/tasks_t36.jpg' },
    { id: 't37', name: 'Task 37: Fix Bug', description: 'Detailed description for task 37. Needs attention.', project_id: 'p19', section_id: 'p19-s1', assignee_id: 'u1', priority: 64, due_date: '2025-08-20T00:00:00.000Z', completed: false, image: '/images/tasks_t37.jpg' },
    { id: 't38', name: 'Task 38: Write Docs', description: 'Detailed description for task 38. Needs attention.', project_id: 'p19', section_id: 'p19-s2', assignee_id: 'u1', priority: 63, due_date: '2025-08-27T00:00:00.000Z', completed: false, image: '/images/tasks_t38.jpg' },
    { id: 't39', name: 'Task 39: Design Icon', description: 'Detailed description for task 39. Needs attention.', project_id: 'p20', section_id: 'p20-s1', assignee_id: 'u1', priority: 62, due_date: '2025-09-03T00:00:00.000Z', completed: false, image: '/images/tasks_t39.jpg' },
    { id: 't40', name: 'Task 40: Meeting', description: 'Detailed description for task 40. Needs attention.', project_id: 'p20', section_id: 'p20-s2', assignee_id: 'u1', priority: 61, due_date: '2025-09-10T00:00:00.000Z', completed: true, image: '/images/tasks_t40.jpg' },
])

// Composition API format - comments data
const comments = ref([
  { id: 'c1', task_id: 't1', user_id: 'u2', text: 'Looking into this.', created_at: '2026-01-07T00:00:00.000Z' }
])

export const useDataStore = defineStore('data', {
  state: () => ({
    users: [...users.value],
    projects: [...projects.value],
    sections: [...sections.value],
    tasks: [...tasks.value],
    comments: [...comments.value],
  }),
  actions: {
    initializeMockData() {
      // Reset to static data
      this.users = [...users.value]
      this.projects = [...projects.value]
      this.sections = [...sections.value]
      this.tasks = [...tasks.value]
      this.comments = [...comments.value]
    },
    getProjects() {
      return this.projects
    },
    addTask(task) {
      this.tasks.push(task)
    },
    addProject(project) {
      this.projects.push(project)
    },
    addSection(section) {
      this.sections.push(section)
    },
    addComment(comment) {
      this.comments.push(comment)
    }
  },
  persist: {
    enabled: false,
  },
})
