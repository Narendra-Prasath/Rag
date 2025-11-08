import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { Pinecone } from "@pinecone-database/pinecone";
import { ChatOpenAI } from "@langchain/openai";


async function processDocument(documentText: string) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  
  const chunks = await splitter.splitText(documentText);
  console.log(`Created ${chunks.length} chunks`);
  return chunks;
}

async function generateEmbeddings(chunks: string[]) {

  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: "sentence-transformers/all-MiniLM-L6-v2",
  });
  
  const vectors = await embeddings.embedDocuments(chunks);
  console.log(`Generated ${vectors.length} vectors`);
  console.log(`Each vector has ${vectors[0].length} dimensions`);
  return vectors;
}

async function storeInPinecone(chunks: string[], vectors: number[][]) {
  const pinecone = new Pinecone({ apiKey: "pcsk_92QWZ_KHYuRB87ZmPByg2ngyd7dQP5ZZcoo7n5jwHBmuPVQDvwUxn6eSvnneUt2ehdj61" });
  const index = pinecone.index("rag-example-768");
  
  const records = chunks.map((chunk, i) => ({
    id: `chunk_${i}`,
    values: vectors[i],
    metadata: { text: chunk }
  }));
  
  await index.upsert(records);
  console.log(`Stored ${records.length} records in Pinecone`);
}

async function queryDocument(question: string) {
  // 1. Embed the question
  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: "sentence-transformers/all-MiniLM-L6-v2",
  });
  const queryVector = await embeddings.embedQuery(question);
  
  // 2. Search Pinecone
  const pinecone = new Pinecone({ apiKey: "pcsk_92QWZ_KHYuRB87ZmPByg2ngyd7dQP5ZZcoo7n5jwHBmuPVQDvwUxn6eSvnneUt2ehdj61" });
  const index = pinecone.index("rag-example-768");
  
  const searchResults = await index.query({
    vector: queryVector,
    topK: 3,
    includeMetadata: true,
  });
  
  // 3. Extract relevant chunks
  const relevantChunks = searchResults.matches.map(match =>
    match.metadata?.text || ""
  );
  
  return relevantChunks;
}

async function generateAnswer(question: string, context: string[]) {
 const llm = new ChatOpenAI({
 apiKey: process.env.OPENAI_API_KEY,
 model: "gpt-4",
 temperature: 0.7,
 });
 
 const prompt = `You are a helpful assistant. Answer the question using only the provided context.
Question: ${question}
Context:
${context.map((c, i) => `[${i + 1}] ${c}`).join("\n\n")}
Answer with citations:`;
 
 const response = await llm.invoke(prompt);
 return response.content;
}

async function answerQuestion(question: string) {
 const relevantChunks = await queryDocument(question);
 const answer = "";//await generateAnswer(question, relevantChunks);
 return answer;
}

// // Example usage
// const doc = "AI is transforming industries... [5000 words]";
// const chunks = await processDocument(doc);
// const vectors = await generateEmbeddings(chunks);
// await storeInPinecone(chunks, vectors);

// const answer = await answerQuestion("What is machine learning?");
// console.log(answer);