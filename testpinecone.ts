import 'dotenv/config';
import * as Pinecone from "@pinecone-database/pinecone";

async function testPinecone() {
  const pinecone = new (Pinecone as any).PineconeClient();
  await pinecone.init({
    apiKey: process.env.PINECONE_API_KEY!,
    environment: process.env.PINECONE_ENVIRONMENT!,
  });

  const index = pinecone.Index("rag-example-768");

  const stats = await index.describeIndexStats();
  console.log("✅ Connected to index successfully");
  console.log(stats);
}

testPinecone().catch(err => console.error("❌ Pinecone error:", err));