import { defineStore } from 'pinia'

export const useDataStore = defineStore('data', {
  state: () => ({
    // Entities
    users: [],
    projects: [],
    sections: [],
    tasks: [],
    comments: [],
  }),
  actions: {
    initializeMockData() {
      // Always rebuild the deterministic mock dataset so selectors like data-id-p1 / data-id-t1 exist
      this.users = []
      this.projects = []
      this.sections = []
      this.tasks = []
      this.comments = []

      // Users
      this.users = [
        { id: 'u1', name: 'Me', avatar: '/images/User.jpg' },
        { id: 'u2', name: 'Alice Smith', avatar: '/images/user-2.jpg' },
        { id: 'u3', name: 'Bob Johnson', avatar: '/images/User.jpg' },
        { id: 'u4', name: 'Charlie Brown', avatar: '/images/User.jpg' },
        { id: 'u5', name: 'Diana Prince', avatar: '/images/DianaPrince.jpg' },
        { id: 'u6', name: 'Evan Wright', avatar: '/images/User.jpg' },
        { id: 'u7', name: 'Fiona Gallagher', avatar: '/images/FionaGallagher.jpg' },
        { id: 'u8', name: 'George Lucas', avatar: '/images/GeorgeLucas.jpg' },
      ]

      // Projects (20 items) – keep priorities high so filters never empty the view
      const projectStatuses = ['Active'] // keep all projects visible when "Active Only" filter is checked
      const sectionTemplates = ['To Do', 'In Progress', 'Done']
      const sectionIdsByProject = {}

      for (let i = 1; i <= 20; i++) {
        const projectId = `p${i}`
        const nameOverride = i === 1 ? 'Project 1: Mobile App' : null
        this.projects.push({
          id: projectId,
          name: nameOverride || `Project ${i}: ${['Website Redesign', 'Mobile App', 'Marketing Campaign', 'Q3 Roadmap', 'Audit', 'Hiring Plan'][i % 6]}`,
          description: `Description for project ${i}. This involves multiple stakeholders.`,
          status: projectStatuses[0],
          priority: 100,
          due_date: new Date(2025, i % 12, (i % 28) + 1).toISOString(),
          owner_id: this.users[i % this.users.length].id,
          image: `/images/project-${i}.jpg`
        })

        // Create a consistent set of sections for every project
        sectionIdsByProject[projectId] = sectionTemplates.map((name, idx) => {
          const sectionId = `${projectId}-s${idx + 1}`
          this.sections.push({
            id: sectionId,
            name,
            project_id: projectId
          })
          return sectionId
        })
      }

      // Tasks (40 items) – every project gets tasks assigned to the current user so filtered views never go empty
      const taskLabels = ['Fix Bug', 'Write Docs', 'Design Icon', 'Meeting', 'Email Client', 'Deploy Server']
      const today = new Date()
      let taskCounter = 1
      for (const project of this.projects) {
        const sectionIds = sectionIdsByProject[project.id]
        for (let j = 0; j < 2; j++) {
          const taskId = `t${taskCounter}`
          const dueDate =
            taskCounter <= 5
              ? today // ensure filterToday has visible tasks (t1–t5)
              : new Date(2025, taskCounter % 12, (taskCounter % 28) + 1)
          // Align task 2 name with automation expectation ("Task 2: Write Docs")
          const nameOverride = taskCounter === 2 ? 'Task 2: Write Docs' : null
          this.tasks.push({
            id: taskId,
            name: nameOverride || `Task ${taskCounter}: ${taskLabels[taskCounter % taskLabels.length]}`,
            description: `Detailed description for task ${taskCounter}. Needs attention.`,
            project_id: project.id,
            section_id: sectionIds[j % sectionIds.length],
            assignee_id: 'u1',
            priority: 100,
            due_date: dueDate.toISOString(),
            completed: taskCounter % 5 === 0,
            image: `/images/task-${(taskCounter % 10) + 1}.jpg`
          })
          taskCounter++
        }
      }
      
      // Comments
      this.comments = [
         { id: 'c1', task_id: 't1', user_id: 'u2', text: 'Looking into this.', created_at: new Date().toISOString() }
      ]
    },
    // Helper to get projects
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
    // Disable persistence to guarantee fresh mock data every app load (avoids stale session data breaking selectors)
    enabled: false,
  },
})

