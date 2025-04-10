import { OpenAI } from '@langchain/openai'
import { PromptTemplate } from '@langchain/core/prompts'
import { LLMChain } from 'langchain/chains'
import { z } from "zod";

import { sendMail } from "@/lib/email";
import { sendSMS } from "@/lib/sms";
import { prisma } from '@/server/db';

// Define types for job search results
export interface JobSearchResult {
  jobTitle: string;
  company: string;
  location: string;
  description: string;
  requirements: string[];
  salary?: string;
  applicationUrl?: string;
  confidenceScore: number;
}

// Initialize the OpenAI model
const model = new OpenAI({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: "gpt-4o",
  temperature: 0.7,
});

// Job search prompt template
const jobSearchPromptTemplate = new PromptTemplate({
  template: `
  You are an expert job hunting assistant. Based on the user's skills and experience, find relevant job opportunities.

  User Skills:
  {skills}

  User Experience:
  {experience}

  Your task is to generate a list of 5 potential job opportunities that match the user's profile.
  For each job, provide:
  - Job title
  - Company name
  - Location (can be remote)
  - Brief job description
  - Key requirements (as a list)
  - Estimated salary range (if possible)
  - Application URL (use a placeholder if unknown)
  - Confidence score (0-1) indicating how well the job matches the user's profile

  Format the output as a valid JSON array of job objects.
  `,
  inputVariables: ["skills", "experience"],
});

// Resume distribution prompt template
const resumeDistributionPromptTemplate = new PromptTemplate({
  template: `
  You are an expert career advisor helping to create personalized messages to send along with a resume.

  Job Details:
  Title: {jobTitle}
  Company: {company}
  Description: {jobDescription}

  User Profile:
  Name: {userName}
  Skills: {userSkills}
  Experience: {userExperience}

  Create a personalized cover letter to send to this company along with the resume. The cover letter should:
  1. Address the company specifically
  2. Highlight the user's relevant skills and experience
  3. Explain why they're a good fit for this position
  4. Be professional, enthusiastic, and concise (around 250 words)
  5. Include a call to action for an interview

  Return only the cover letter text without any additional commentary.
  `,
  inputVariables: ["jobTitle", "company", "jobDescription", "userName", "userSkills", "userExperience"],
});

// Schema for job search input
export const jobSearchSchema = z.object({
  userId: z.string(),
  jobTitle: z.string().optional(),
  location: z.string().optional(),
  remote: z.boolean().optional(),
});

// Schema for resume distribution input
export const resumeDistributionSchema = z.object({
  userId: z.string(),
  jobAlertId: z.string().optional(),
  companyName: z.string(),
  jobTitle: z.string(),
  contactEmail: z.string().email(),
  contactPerson: z.string().optional(),
  jobDescription: z.string(),
});

/**
 * Search for relevant jobs based on user skills and experience
 */
export async function searchJobs(input: z.infer<typeof jobSearchSchema>): Promise<JobSearchResult[]> {
  try {
    const { userId, jobTitle, location, remote } = input;

    // Get user skills and experience
    const userSkills = await prisma.skill.findMany({
      where: { userId },
      select: {
        name: true,
        category: true,
        proficiency: true,
        yearsOfExperience: true,
      },
    });

    const userExperience = await prisma.experience.findMany({
      where: { userId },
      select: {
        company: true,
        position: true,
        description: true,
        startDate: true,
        endDate: true,
        achievements: true,
      },
    });

    if (!userSkills.length || !userExperience.length) {
      throw new Error("User must have skills and experience records to search for jobs");
    }

    // Format the skills and experience data
    const skillsText = userSkills
      .map((skill) => `${skill.name} (${skill.category}, ${skill.proficiency}%, ${skill.yearsOfExperience} years)`)
      .join("\n");

    const experienceText = userExperience
      .map((exp) => {
        const duration = exp.endDate
          ? `${exp.startDate.toISOString().split("T")[0]} to ${exp.endDate.toISOString().split("T")[0]}`
          : `${exp.startDate.toISOString().split("T")[0]} to Present`;

        return `${exp.position} at ${exp.company} (${duration})\n${exp.description}\nAchievements: ${exp.achievements.join(", ")}`;
      })
      .join("\n\n");

    // Add search filters if provided
    let searchFilters = "";
    if (jobTitle) {
      searchFilters += `\nAdditional search criteria:\n- Job Title: ${jobTitle}`;
    }
    if (location) {
      searchFilters += `\n- Location: ${location}`;
    }
    if (remote !== undefined) {
      searchFilters += `\n- Remote: ${remote ? "Yes" : "No"}`;
    }

    // Create the LLM chain
    const jobSearchChain = new LLMChain({
      llm: model,
      prompt: jobSearchPromptTemplate,
    });

    // Execute the chain
    const result = await jobSearchChain.call({
      skills: skillsText + searchFilters,
      experience: experienceText,
    });

    // Parse the results (the AI returns a JSON string)
    const jobResults: JobSearchResult[] = JSON.parse(
      result.text.replace(/```json|```/g, "").trim()
    );

    // Save the job alerts to the database
    for (const job of jobResults) {
      await prisma.jobAlert.create({
        data: {
          userId,
          jobTitle: job.jobTitle,
          company: job.company,
          location: job.location || "Remote",
          description: job.description,
          requirements: job.requirements,
          salary: job.salary,
          applicationUrl: job.applicationUrl,
          confidence: job.confidenceScore,
        },
      });
    }

    // Send notification to the user
    const userEmail = process.env.USER_EMAIL;
    const userPhone = process.env.USER_PHONE;

    if (userEmail) {
      await sendMail({
        to: userEmail,
        subject: `New Job Matches Found: ${jobResults.length} opportunities`,
        text: `The AI agent found ${jobResults.length} new job opportunities matching your profile. Log in to your portfolio dashboard to view them.`,
      });
    }

    if (userPhone) {
      await sendSMS({
        to: userPhone,
        body: `Your portfolio AI found ${jobResults.length} new job opportunities matching your skills. Check your dashboard for details.`,
      });
    }

    return jobResults;
  } catch (error) {
    console.error("Error in job search agent:", error);
    throw new Error("Failed to search for jobs: " + (error instanceof Error ? error.message : "Unknown error"));
  }
}

/**
 * Create and distribute resume with a personalized cover letter
 */
export async function distributeResume(input: z.infer<typeof resumeDistributionSchema>): Promise<{ success: boolean; message: string }> {
  try {
    const { userId, jobAlertId, companyName, jobTitle, contactEmail, contactPerson, jobDescription } = input;

    // Get user profile information
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { name: true },
    });

    if (!user) {
      throw new Error("User not found");
    }

    // Get user skills and experience
    const userSkills = await prisma.skill.findMany({
      where: { userId },
      select: { name: true, proficiency: true },
      orderBy: { proficiency: "desc" },
      take: 10, // Get top 10 skills by proficiency
    });

    const userExperience = await prisma.experience.findMany({
      where: { userId },
      select: { position: true, company: true, description: true },
      orderBy: { startDate: "desc" },
      take: 3, // Get 3 most recent experiences
    });

    // Format the skills and experience
    const skillsText = userSkills.map((skill) => skill.name).join(", ");

    const experienceText = userExperience
      .map((exp) => `${exp.position} at ${exp.company}: ${exp.description}`)
      .join("\n\n");

    // Create the LLM chain for cover letter generation
    const coverLetterChain = new LLMChain({
      llm: model,
      prompt: resumeDistributionPromptTemplate,
    });

    // Generate the cover letter
    const result = await coverLetterChain.call({
      jobTitle,
      company: companyName,
      jobDescription,
      userName: user.name,
      userSkills: skillsText,
      userExperience: experienceText,
    });

    const coverLetter = result.text.trim();

    // Record the resume distribution in the database
    const distribution = await prisma.resumeDistribution.create({
      data: {
        companyName,
        contactEmail,
        contactPerson: contactPerson || null,
        status: "SENT",
      },
    });

    // If this was from a job alert, update the job alert status
    if (jobAlertId) {
      await prisma.jobAlert.update({
        where: { id: jobAlertId },
        data: { status: "APPLIED", appliedAt: new Date() },
      });
    }

    // In a real implementation, we would send the actual email with resume attached
    // For this demo, we'll simulate the email sending
    await sendMail({
      to: contactEmail,
      subject: `Application for ${jobTitle} position at ${companyName}`,
      text: coverLetter,
      // We would attach the resume here
    });

    // Send notification to the user
    const userEmail = process.env.USER_EMAIL;
    if (userEmail) {
      await sendMail({
        to: userEmail,
        subject: `Resume sent to ${companyName}`,
        text: `Your resume has been sent to ${companyName} for the ${jobTitle} position. A follow-up has been scheduled for one week from today.`,
      });
    }

    return {
      success: true,
      message: `Resume and cover letter sent to ${companyName} for the ${jobTitle} position.`,
    };
  } catch (error) {
    console.error("Error in resume distribution agent:", error);
    return {
      success: false,
      message: "Failed to distribute resume: " + (error instanceof Error ? error.message : "Unknown error"),
    };
  }
}
