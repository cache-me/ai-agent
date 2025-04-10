import { OpenAI } from '@langchain/openai'
import { PromptTemplate } from '@langchain/core/prompts'
import { LLMChain } from 'langchain/chains'
import { z } from 'zod'
import { prisma } from '@/server/db'

// Initialize the OpenAI model
const model = new OpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: 'gpt-4o',
  temperature: 0.7,
})

// Define schema for chat interactions
export const chatMessageSchema = z.object({
  content: z.string().min(1, 'Message cannot be empty'),
  chatId: z.string().optional(),
  visitorName: z.string().optional(),
  visitorEmail: z.string().email().optional(),
})

// Define types for chat response
export interface ChatResponse {
  content: string
  chatId: string
  messageId: string
}

// Initial system prompt with portfolio owner information
const portfolioOwnerInfoPrompt = async () => {
  // Get the portfolio owner's information (assuming admin user)
  const admin = await prisma.user.findFirst({
    where: { role: 'ADMIN' },
    include: {
      skills: true,
      education: true,
      experiences: true,
      projects: true,
    },
  })

  if (!admin) {
    return 'You are a helpful assistant for a portfolio website.'
  }

  // Format skills
  const skills = admin.skills
    .map((skill) => `${skill.name} (${skill.proficiency}% proficiency, ${skill.yearsOfExperience} years)`)
    .join(', ')

  // Format education
  const education = admin.education
    .map((edu) => {
      const startDate = edu.startDate.getFullYear()
      const endDate = edu.endDate ? edu.endDate.getFullYear() : 'Present'
      return `${edu.degree} in ${edu.fieldOfStudy} from ${edu.institution} (${startDate}-${endDate})`
    })
    .join('; ')

  // Format experience
  const experience = admin.experiences
    .map((exp) => {
      const startDate = exp.startDate.getFullYear()
      const endDate = exp.endDate ? exp.endDate.getFullYear() : 'Present'
      return `${exp.position} at ${exp.company} (${startDate}-${endDate})`
    })
    .join('; ')

  // Format projects
  const projects = admin.projects.map((project) => `${project.title}: ${project.description}`).join('; ')

  return `
  You are an AI assistant representing ${admin.name}, a professional in the tech industry.

  About ${admin.name}:
  ${admin.bio || ''}

  Skills: ${skills}

  Education: ${education}

  Experience: ${experience}

  Notable Projects: ${projects}

  Your role is to engage with visitors to ${admin.name}'s portfolio website, answer questions about ${
    admin.name
  }'s background, skills, experience, and projects.
  You can also discuss potential collaborations, job opportunities, or freelance work.

  Be professional, friendly, and helpful. If you don't know specific details that weren't provided above,
  you can suggest the visitor to contact ${admin.name} directly for more information.

  You should NOT make up information about ${admin.name} that wasn't provided to you.
  If asked about something not covered in the information above, politely explain that you don't have that specific information.
  `
}

// Chat prompt template
const chatPromptTemplate = new PromptTemplate({
  template: `
  {systemPrompt}

  Chat History:
  {chatHistory}

  Visitor: {userMessage}

  Assistant:`,
  inputVariables: ['systemPrompt', 'chatHistory', 'userMessage'],
})

/**
 * Process a chat message and generate a response
 */
export async function processChatMessage(input: z.infer<typeof chatMessageSchema>): Promise<ChatResponse> {
  try {
    const { content, chatId, visitorName, visitorEmail } = input

    // Get or create chat session
    let chat
    if (chatId) {
      // Get existing chat
      chat = await prisma.chatInteraction.findUnique({
        where: { id: chatId },
        include: { conversation: true },
      })

      if (!chat) {
        throw new Error(`Chat session not found: ${chatId}`)
      }
    } else {
      // Create new chat
      chat = await prisma.chatInteraction.create({
        data: {
          visitorName: visitorName || 'Anonymous',
          visitorEmail,
          conversation: {
            create: [],
          },
        },
        include: { conversation: true },
      })
    }

    // Get system prompt with portfolio owner info
    const systemPrompt = await portfolioOwnerInfoPrompt()

    // Format chat history
    const chatHistory = chat.conversation
      .map((msg) => `${msg.sender === 'USER' ? 'Visitor' : 'Assistant'}: ${msg.content}`)
      .join('\n')

    // Add user message to the database
    await prisma.chatMessage.create({
      data: {
        content,
        sender: 'USER',
        chatId: chat.id,
      },
    })

    // Create the LLM chain
    const chatChain = new LLMChain({
      llm: model,
      prompt: chatPromptTemplate,
    })

    // Generate the response
    const result = await chatChain.call({
      systemPrompt,
      chatHistory,
      userMessage: content,
    })

    const aiResponse = result.text.trim()

    // Save AI response to the database
    const message = await prisma.chatMessage.create({
      data: {
        content: aiResponse,
        sender: 'AI',
        chatId: chat.id,
      },
    })

    return {
      content: aiResponse,
      chatId: chat.id,
      messageId: message.id,
    }
  } catch (error) {
    console.error('Error in chat agent:', error)
    throw new Error('Failed to process chat message: ' + (error instanceof Error ? error.message : 'Unknown error'))
  }
}

/**
 * Get chat history
 */
export async function getChatHistory(chatId: string): Promise<{
  chat: any
  messages: any[]
}> {
  try {
    const chat = await prisma.chatInteraction.findUnique({
      where: { id: chatId },
    })

    if (!chat) {
      throw new Error(`Chat session not found: ${chatId}`)
    }

    const messages = await prisma.chatMessage.findMany({
      where: { chatId },
      orderBy: { timestamp: 'asc' },
    })

    return { chat, messages }
  } catch (error) {
    console.error('Error getting chat history:', error)
    throw new Error('Failed to get chat history: ' + (error instanceof Error ? error.message : 'Unknown error'))
  }
}
