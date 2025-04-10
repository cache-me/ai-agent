import nodemailer from 'nodemailer'

interface EmailParams {
  to: string
  subject: string
  text: string
  html?: string
  attachments?: Array<{
    filename: string
    path: string
    contentType?: string
  }>
}

export async function sendMail({ to, subject, text, html, attachments }: EmailParams): Promise<boolean> {
  try {
    // Create a transporter object
    const transporter = nodemailer.createTransport({
      host: process.env.EMAIL_SERVER_HOST,
      port: Number(process.env.EMAIL_SERVER_PORT),
      secure: Number(process.env.EMAIL_SERVER_PORT) === 465, // true for 465, false for other ports
      auth: {
        user: process.env.EMAIL_SERVER_USER,
        pass: process.env.EMAIL_SERVER_PASSWORD,
      },
    })

    // Send the email
    const info = await transporter.sendMail({
      from: process.env.EMAIL_FROM,
      to,
      subject,
      text,
      html: html || text.replace(/\n/g, '<br>'),
      attachments,
    })

    console.log('Email sent:', info.messageId)
    return true
  } catch (error) {
    console.error('Error sending email:', error)
    // In development, we don't want to fail if email sending fails
    if (process.env.NODE_ENV === 'development') {
      console.log('Email would have been sent (dev mode):', { to, subject, text })
      return true
    }
    return false
  }
}
