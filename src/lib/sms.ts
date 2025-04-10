import twilio from 'twilio';

interface SMSParams {
  to: string;
  body: string;
}

export async function sendSMS({ to, body }: SMSParams): Promise<boolean> {
  try {
    const accountSid = process.env.TWILIO_ACCOUNT_SID;
    const authToken = process.env.TWILIO_AUTH_TOKEN;
    const from = process.env.TWILIO_PHONE_NUMBER;

    if (!accountSid || !authToken || !from) {
      console.warn('Missing Twilio credentials. SMS not sent.');
      return false;
    }

    const client = twilio(accountSid, authToken);

    const message = await client.messages.create({
      body,
      from,
      to,
    });

    console.log('SMS sent:', message.sid);
    return true;
  } catch (error) {
    console.error('Error sending SMS:', error);
    // In development, we don't want to fail if SMS sending fails
    if (process.env.NODE_ENV === 'development') {
      console.log('SMS would have been sent (dev mode):', { to, body });
      return true;
    }
    return false;
  }
}
