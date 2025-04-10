import { OpenAI } from '@langchain/openai'
import { PromptTemplate } from '@langchain/core/prompts'
import { LLMChain } from 'langchain/chains'
import { z } from 'zod'
import { sendMail } from '@/lib/email'
import { sendSMS } from '@/lib/sms'
import { prisma } from '@/server/db'

// Initialize the OpenAI model
const model = new OpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: 'gpt-4o',
  temperature: 0.4,
})

// Schema for content analysis input
export const portfolioAnalysisSchema = z.object({
  userId: z.string(),
})

// Schema for skill update suggestion input
export const skillUpdateSchema = z.object({
  userId: z.string(),
  skillAreaFocus: z
    .enum(['FRONTEND', 'BACKEND', 'DATABASE', 'DEVOPS', 'DESIGN', 'AI_ML', 'MOBILE', 'OTHER'])
    .optional(),
})

// Schema for creating a portfolio reminder
export const reminderCreateSchema = z.object({
  title: z.string(),
  description: z.string(),
  dueDate: z.date(),
  priority: z.enum(['LOW', 'MEDIUM', 'HIGH', 'URGENT']).default('MEDIUM'),
  category: z.enum(['SKILL_UPDATE', 'PROJECT_ADD', 'RESUME_UPDATE', 'EXPERIENCE_UPDATE', 'CONTENT_REFRESH', 'OTHER']),
})

// Content analysis prompt template
const contentAnalysisPromptTemplate = new PromptTemplate({
  template: `
  You are an AI portfolio content manager. Your job is to analyze the user's portfolio content and provide recommendations for improvements.

  Skills:
  {skills}

  Projects:
  {projects}

  Experience:
  {experience}

  Education:
  {education}

  Based on this information, analyze the following aspects:
  1. Content gaps and inconsistencies
  2. Skills that should be highlighted more prominently
  3. Projects that would showcase the user's abilities better
  4. Areas where the portfolio could be improved for better job prospects
  5. Trends in the industry that the user should consider adding to their portfolio

  Provide a detailed analysis with specific recommendations. Format your response as a JSON object with these keys:
  - contentGaps: Array of specific content gaps
  - skillHighlights: Array of skills that should be highlighted
  - projectSuggestions: Array of project ideas to showcase skills
  - improvementAreas: Array of areas to improve
  - trendSuggestions: Array of industry trends to consider
  `,
  inputVariables: ['skills', 'projects', 'experience', 'education'],
})

// Skill update suggestion prompt template
const skillUpdatePromptTemplate = new PromptTemplate({
  template: `
  You are an AI career development advisor. Your job is to analyze the user's current skills and suggest updates based on current tech trends and job market demands.

  Current User Skills:
  {skills}

  Current User Projects:
  {projects}

  Current Experience:
  {experience}

  Current Tech Trends:
  {techTrends}

  Focus Area: {focusArea}

  Based on this information, suggest new skills or updates to existing skills that would make the user more competitive in the job market,
  particularly focusing on {focusArea} if specified.

  For each skill suggestion, provide:
  1. Skill name
  2. Category (FRONTEND, BACKEND, DATABASE, etc.)
  3. Why it's important in today's job market
  4. How it complements the user's existing skills
  5. Learning resources (courses, documentation, tutorials)
  6. Estimated time to become proficient

  Format your response as a JSON array of skill suggestion objects.
  `,
  inputVariables: ['skills', 'projects', 'experience', 'techTrends', 'focusArea'],
})

// Reminder creation prompt template
const reminderPromptTemplate = new PromptTemplate({
  template: `
  You are an AI portfolio manager scheduling regular updates for the user's portfolio.

  User's Portfolio Last Updates:
  Skills last updated: {skillsLastUpdated}
  Projects last updated: {projectsLastUpdated}
  Experience last updated: {experienceLastUpdated}

  Current date: {currentDate}

  Based on this information, create a set of reminders for the user to keep their portfolio updated.

  Generate 3-5 specific reminders, each with:
  1. Title (short, specific)
  2. Description (detailed, actionable)
  3. Suggested due date (in YYYY-MM-DD format)
  4. Priority (LOW, MEDIUM, HIGH, URGENT)
  5. Category (SKILL_UPDATE, PROJECT_ADD, RESUME_UPDATE, EXPERIENCE_UPDATE, CONTENT_REFRESH, OTHER)

  Format your response as a JSON array of reminder objects.
  `,
  inputVariables: ['skillsLastUpdated', 'projectsLastUpdated', 'experienceLastUpdated', 'currentDate'],
})

/**
 * Analyze the user's portfolio content and provide improvement recommendations
 */
export async function analyzePortfolioContent(input: z.infer<typeof portfolioAnalysisSchema>): Promise<{
  contentGaps: string[]
  skillHighlights: string[]
  projectSuggestions: string[]
  improvementAreas: string[]
  trendSuggestions: string[]
}> {
  try {
    const { userId } = input

    // Fetch user's portfolio data
    const skills = await prisma.skill.findMany({
      where: { userId },
      select: {
        name: true,
        category: true,
        proficiency: true,
        yearsOfExperience: true,
        description: true,
      },
    })

    const projects = await prisma.project.findMany({
      where: { userId },
      select: {
        title: true,
        description: true,
        category: true,
        skills: {
          include: {
            skill: true,
          },
        },
        completionDate: true,
      },
    })

    const experience = await prisma.experience.findMany({
      where: { userId },
      select: {
        company: true,
        position: true,
        description: true,
        startDate: true,
        endDate: true,
        achievements: true,
      },
    })

    const education = await prisma.education.findMany({
      where: { userId },
      select: {
        institution: true,
        degree: true,
        fieldOfStudy: true,
        startDate: true,
        endDate: true,
        description: true,
      },
    })

    // Format the data for the prompt
    const skillsText = JSON.stringify(skills)
    const projectsText = JSON.stringify(projects)
    const experienceText = JSON.stringify(experience)
    const educationText = JSON.stringify(education)

    // Create and execute the content analysis chain
    const contentAnalysisChain = new LLMChain({
      llm: model,
      prompt: contentAnalysisPromptTemplate,
    })

    const result = await contentAnalysisChain.call({
      skills: skillsText,
      projects: projectsText,
      experience: experienceText,
      education: educationText,
    })

    // Parse the results
    const analysisResults = JSON.parse(result.text.replace(/```json|```/g, '').trim())

    return analysisResults
  } catch (error) {
    console.error('Error in portfolio content analysis:', error)
    throw new Error(
      'Failed to analyze portfolio content: ' + (error instanceof Error ? error.message : 'Unknown error'),
    )
  }
}

/**
 * Suggest skill updates based on current technology trends
 */
export async function suggestSkillUpdates(input: z.infer<typeof skillUpdateSchema>): Promise<
  Array<{
    name: string
    category: string
    importance: string
    complementsExisting: string
    learningResources: string[]
    estimatedTimeToLearn: string
  }>
> {
  try {
    const { userId, skillAreaFocus } = input

    // Fetch user's skills, projects, and experience
    const skills = await prisma.skill.findMany({
      where: { userId },
      select: {
        name: true,
        category: true,
        proficiency: true,
        yearsOfExperience: true,
      },
    })

    const projects = await prisma.project.findMany({
      where: { userId },
      select: {
        title: true,
        description: true,
        category: true,
      },
    })

    const experience = await prisma.experience.findMany({
      where: { userId },
      select: {
        position: true,
        description: true,
      },
    })

    // Fetch current technology trends
    const techTrends = await prisma.technologyTrend.findMany({
      select: {
        name: true,
        category: true,
        popularityScore: true,
        growthRate: true,
        description: true,
      },
      orderBy: { growthRate: 'desc' },
      take: 10,
    })

    // Format the data for the prompt
    const skillsText = JSON.stringify(skills)
    const projectsText = JSON.stringify(projects)
    const experienceText = JSON.stringify(experience)
    const techTrendsText = JSON.stringify(techTrends)
    const focusArea = skillAreaFocus || 'all skill areas'

    // Create and execute the skill update suggestion chain
    const skillUpdateChain = new LLMChain({
      llm: model,
      prompt: skillUpdatePromptTemplate,
    })

    const result = await skillUpdateChain.call({
      skills: skillsText,
      projects: projectsText,
      experience: experienceText,
      techTrends: techTrendsText,
      focusArea,
    })

    // Parse the results
    const skillSuggestions = JSON.parse(result.text.replace(/```json|```/g, '').trim())

    return skillSuggestions
  } catch (error) {
    console.error('Error in skill update suggestions:', error)
    throw new Error('Failed to suggest skill updates: ' + (error instanceof Error ? error.message : 'Unknown error'))
  }
}

/**
 * Generate portfolio update reminders
 */
export async function generateReminders(userId: string): Promise<z.infer<typeof reminderCreateSchema>[]> {
  try {
    // Get last update timestamps for different portfolio areas
    const skills = await prisma.skill.findMany({
      where: { userId },
      orderBy: { updatedAt: 'desc' },
      take: 1,
      select: { updatedAt: true },
    })

    const projects = await prisma.project.findMany({
      where: { userId },
      orderBy: { updatedAt: 'desc' },
      take: 1,
      select: { updatedAt: true },
    })

    const experience = await prisma.experience.findMany({
      where: { userId },
      orderBy: { updatedAt: 'desc' },
      take: 1,
      select: { updatedAt: true },
    })

    // Format dates
    const skillsLastUpdated = skills.length > 0 ? skills[0].updatedAt.toISOString().split('T')[0] : 'never'

    const projectsLastUpdated = projects.length > 0 ? projects[0].updatedAt.toISOString().split('T')[0] : 'never'

    const experienceLastUpdated = experience.length > 0 ? experience[0].updatedAt.toISOString().split('T')[0] : 'never'

    const currentDate = new Date().toISOString().split('T')[0]

    // Create and execute the reminder generation chain
    const reminderChain = new LLMChain({
      llm: model,
      prompt: reminderPromptTemplate,
    })

    const result = await reminderChain.call({
      skillsLastUpdated,
      projectsLastUpdated,
      experienceLastUpdated,
      currentDate,
    })

    // Parse the results
    const reminderSuggestions = JSON.parse(result.text.replace(/```json|```/g, '').trim())

    // Create the reminders in the database
    const createdReminders = []
    for (const reminder of reminderSuggestions) {
      const createdReminder = await prisma.portfolioReminder.create({
        data: {
          title: reminder.title,
          description: reminder.description,
          dueDate: new Date(reminder.dueDate),
          priority: reminder.priority,
          category: reminder.category,
          completed: false,
          notificationSent: false,
        },
      })
      createdReminders.push(createdReminder)
    }

    // Send a notification about the new reminders
    const userEmail = process.env.USER_EMAIL
    if (userEmail) {
      await sendMail({
        to: userEmail,
        subject: 'New Portfolio Update Reminders',
        text: `Your AI portfolio manager has created ${createdReminders.length} new reminders to help you keep your portfolio up-to-date. Check your dashboard to view them.`,
      })
    }

    return createdReminders
  } catch (error) {
    console.error('Error generating portfolio reminders:', error)
    throw new Error('Failed to generate reminders: ' + (error instanceof Error ? error.message : 'Unknown error'))
  }
}

/**
 * Send reminder notifications for due portfolio updates
 */
export async function sendReminderNotifications(): Promise<number> {
  try {
    const now = new Date()

    // Find reminders due within the next 48 hours that haven't had notifications sent yet
    const dueDate = new Date(now)
    dueDate.setHours(now.getHours() + 48)

    const dueReminders = await prisma.portfolioReminder.findMany({
      where: {
        dueDate: { lte: dueDate },
        completed: false,
        notificationSent: false,
      },
    })

    if (dueReminders.length === 0) {
      return 0
    }

    // Send notifications for each reminder
    let notificationsSent = 0
    const userEmail = process.env.USER_EMAIL
    const userPhone = process.env.USER_PHONE

    for (const reminder of dueReminders) {
      const dueFormatted = reminder.dueDate.toLocaleDateString()
      const message = `REMINDER: ${reminder.title} - Due ${dueFormatted}.\n${reminder.description}`

      // Send email notification
      if (userEmail) {
        await sendMail({
          to: userEmail,
          subject: `Portfolio Reminder: ${reminder.title}`,
          text: message,
        })
      }

      // Send SMS for high priority reminders
      if (userPhone && (reminder.priority === 'HIGH' || reminder.priority === 'URGENT')) {
        await sendSMS({
          to: userPhone,
          body: `${reminder.priority} PRIORITY: ${reminder.title} due on ${dueFormatted}`,
        })
      }

      // Mark the reminder as notified
      await prisma.portfolioReminder.update({
        where: { id: reminder.id },
        data: { notificationSent: true },
      })

      notificationsSent++
    }

    return notificationsSent
  } catch (error) {
    console.error('Error sending reminder notifications:', error)
    return 0
  }
}
