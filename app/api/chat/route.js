import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from 'openai';

// Define the system prompt to guide the AI's behavior
const systemPrompt = `
You are a Rate My Professor assistant to help students find professors and courses.
For every user question, return the top 3 professors that match the query, and answer using relevant reviews.
Please format the information for each professor as follows:
{
  "professor": str,
  "review": str,
  "subject": str,
  "stars": float
}
If there are multiple reviews for the same professor, make sure to combine them into one entry with just one review.
After providing the information in JSON format, provide additional helpful insights in a conversational manner, as if speaking to a student. Make sure your conversational tone is friendly and informative.
`;

// Create the POST function to handle incoming requests
export async function POST(req) {
  const data = await req.json();

  // Initialize Pinecone and OpenAI
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const index = pc.index('rag').namespace('ns1');
  const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });

  // Extract the user query and create an embedding
  const text = data[data.length - 1].content;
  const embedding = await openai.embeddings.create({
    model: 'text-embedding-ada-002', // Choose the embedding model
    input: text,
  });

  // Query Pinecone for the top 3 matching results
  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  // Format the results into a readable string
  let resultString = '';
  results.matches.forEach((match) => {
    resultString += `
    Professor: ${match.id}
    Review: ${match.metadata.review}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n`;
  });

  // Prepare the content to be sent to OpenAI for a response
  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);

  // Send a chat completion request to OpenAI
  const completion = await openai.chat.completions.create({
    messages: [
      { role: 'system', content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: 'user', content: lastMessageContent },
    ],
    model: 'gpt-3.5-turbo',
    stream: true,
  });

  // Handle the streaming response from OpenAI
  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}
