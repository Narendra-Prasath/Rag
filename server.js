import express from 'express';
import cors from 'cors';
import 'dotenv/config';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { GoogleGenAI } from "@google/genai";

// --- Configuration ---
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = "rag-example-768";
const PORT = process.env.PORT || 3000;
const NODE_ENV = process.env.NODE_ENV || 'development';

// Validate required environment variables
if (!OPENAI_API_KEY || !PINECONE_API_KEY) {
    console.error("❌ FATAL ERROR: OPENAI_API_KEY or PINECONE_API_KEY is missing in the .env file.");
    process.exit(1);
}

// --- RAG Pipeline Functions ---

async function processDocument(documentText) {
    console.log("[RAG] Step 1: Document Processing (Chunking)");
    
    if (!documentText || typeof documentText !== 'string') {
        throw new Error('Invalid document text provided');
    }
    
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 500,
        chunkOverlap: 100,
    });
    
    const chunks = await splitter.splitText(documentText);
    
    if (chunks.length === 0) {
        throw new Error('Document chunking resulted in zero chunks');
    }
    
    console.log(`[RAG] ✅ Created ${chunks.length} chunks.`);
    return chunks;
}

async function generateEmbeddings(chunks) {
    console.log("[RAG] Step 2: Generate Embeddings");

    if (!Array.isArray(chunks) || chunks.length === 0) {
        throw new Error('Invalid chunks array provided');
    }

    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: "text-embedding-004",
    });

    const vectors = await embeddings.embedDocuments(chunks);

    // Validate vectors
    if (!vectors || vectors.length === 0 || vectors[0].length === 0) {
        throw new Error('Failed to generate valid embeddings - empty vectors returned');
    }

    console.log(`[RAG] ✅ Generated ${vectors.length} vectors (dimension: ${vectors[0].length}).`);
    return vectors;
}

async function storeInPinecone(chunks, vectors) {
    console.log("[RAG] Step 3: Store in Vector Database (Pinecone)");
    
    if (chunks.length !== vectors.length) {
        throw new Error('Mismatch between chunks and vectors length');
    }
    
    const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
    const index = pinecone.index(PINECONE_INDEX_NAME);
    
    const records = chunks.map((chunk, i) => ({
        id: `chunk_${i}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        values: vectors[i],
        metadata: { text: chunk, timestamp: new Date().toISOString() }
    }));
    
    // Batch upsert for better performance with large datasets
    const batchSize = 100;
    for (let i = 0; i < records.length; i += batchSize) {
        const batch = records.slice(i, i + batchSize);
        await index.upsert(batch);
    }
    
    console.log(`[RAG] ✅ Stored ${records.length} records in Pinecone.`);
    return records.length;
}

async function queryDocument(question) {
    console.log("[RAG] Step 4: Query and Retrieve");

    if (!question || typeof question !== 'string') {
        throw new Error('Invalid question provided');
    }

    // 1. Embed the question
    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: "text-embedding-004",
    });
    const queryVector = await embeddings.embedQuery(question);
    
    // 2. Search Pinecone
    const pinecone = new Pinecone({ apiKey: PINECONE_API_KEY });
    const index = pinecone.index(PINECONE_INDEX_NAME);
    
    const searchResults = await index.query({
        vector: queryVector,
        topK: 3,
        includeMetadata: true,
    });
    
    // 3. Extract relevant chunks
    const relevantChunks = searchResults.matches
        .filter(match => match.metadata?.text && typeof match.metadata.text === 'string')
        .map(match => match.metadata.text);
        
    console.log(`[RAG] ✅ Retrieved ${relevantChunks.length} relevant chunks.`);
    return relevantChunks;
}

async function generateAnswer(question, context) {
    console.log("[RAG] Step 5: Generate Answer with LLM (RAG)");

    if (!Array.isArray(context) || context.length === 0) {
        throw new Error('No context available for answer generation');
    }

    const ai = new GoogleGenAI({
        apiKey: process.env.GEMINI_API_KEY,
    });

    const prompt = `You are a helpful assistant. Answer the question using ONLY the provided context.
If the answer is not found in the context, state that clearly and do not make up information.

Question: ${question}

Context:
${context.map((c, i) => `[${i + 1}] ${c}`).join("\n\n")}

Answer with citations (e.g., [1], [2]):`;

    const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
    });

    console.log("[RAG] ✅ LLM response generated.");
    return response.text;
}

// --- Express App Setup ---
const app = express();

// Middleware
app.use(express.json({ limit: '10mb' })); // Reduced from 50mb for security
app.use(cors({
    origin: NODE_ENV === 'production' 
        ? process.env.ALLOWED_ORIGINS?.split(',') || []
        : '*',
    methods: ['GET', 'POST'],
    credentials: true
}));

// Request logging middleware
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
    next();
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.status(200).json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        environment: NODE_ENV
    });
});

// API Endpoint for Indexing (Steps 1-3)
app.post('/api/index-document', async (req, res) => {
    const startTime = Date.now();
    
    try {
        const { documentText } = req.body;
        
        if (!documentText) {
            return res.status(400).json({ 
                error: 'Document text is required.',
                success: false 
            });
        }
        
        if (typeof documentText !== 'string') {
            return res.status(400).json({ 
                error: 'Document text must be a string.',
                success: false 
            });
        }
        
        if (documentText.length > 500000) { // 500KB limit
            return res.status(400).json({ 
                error: 'Document text exceeds maximum length of 500,000 characters.',
                success: false 
            });
        }
        
        console.log("\n=== BACKEND: INDEXING STARTED ===");
        
        const chunks = await processDocument(documentText);
        const vectors = await generateEmbeddings(chunks);
        const recordCount = await storeInPinecone(chunks, vectors);
        
        const duration = Date.now() - startTime;
        console.log(`=== BACKEND: INDEXING FINISHED (${duration}ms) ===\n`);
        
        res.status(200).json({ 
            message: `Document successfully indexed with ${chunks.length} chunks.`,
            success: true,
            chunkCount: chunks.length,
            recordCount: recordCount,
            duration: duration
        });
        
    } catch (error) {
        const duration = Date.now() - startTime;
        console.error("❌ BACKEND ERROR - Indexing failed:", error.message);
        console.error(error.stack);
        
        res.status(500).json({ 
            error: 'Failed to index document. Please try again.',
            success: false,
            details: NODE_ENV === 'development' ? error.message : undefined,
            duration: duration
        });
    }
});

// API Endpoint for Querying (Steps 4 & 5)
app.post('/api/answer-question', async (req, res) => {
    const startTime = Date.now();
    
    try {
        const { question } = req.body;
        
        if (!question) {
            return res.status(400).json({ 
                error: 'Question is required.',
                success: false 
            });
        }
        
        if (typeof question !== 'string') {
            return res.status(400).json({ 
                error: 'Question must be a string.',
                success: false 
            });
        }
        
        if (question.length > 1000) {
            return res.status(400).json({ 
                error: 'Question exceeds maximum length of 1,000 characters.',
                success: false 
            });
        }
        
        console.log("\n=== BACKEND: QUERY STARTED ===");
        
        const relevantChunks = await queryDocument(question);
        
        if (relevantChunks.length === 0) {
            const duration = Date.now() - startTime;
            console.log(`=== BACKEND: QUERY FINISHED - No context (${duration}ms) ===\n`);
            
            return res.status(200).json({ 
                answer: "I couldn't find any relevant information in the document store to answer that question.",
                success: true,
                chunksRetrieved: 0,
                duration: duration
            });
        }
        
        const answer = await generateAnswer(question, relevantChunks);
        
        const duration = Date.now() - startTime;
        console.log(`=== BACKEND: QUERY FINISHED (${duration}ms) ===\n`);
        
        res.status(200).json({ 
            answer,
            success: true,
            chunksRetrieved: relevantChunks.length,
            duration: duration
        });
        
    } catch (error) {
        const duration = Date.now() - startTime;
        console.error("❌ BACKEND ERROR - Query failed:", error.message);
        console.error(error.stack);
        
        res.status(500).json({ 
            error: 'Failed to answer the question. Please try again.',
            success: false,
            details: NODE_ENV === 'development' ? error.message : undefined,
            duration: duration
        });
    }
});

// 404 handler
app.use((req, res) => {
    res.status(404).json({ 
        error: 'Endpoint not found',
        success: false 
    });
});

// Global error handler
app.use((err, req, res, next) => {
    console.error("❌ Unhandled error:", err);
    res.status(500).json({ 
        error: 'Internal server error',
        success: false,
        details: NODE_ENV === 'development' ? err.message : undefined
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`\n${"=".repeat(60)}`);
    console.log(`🚀 RAG Backend Server`);
    console.log(`${"=".repeat(60)}`);
    console.log(`Environment: ${NODE_ENV}`);
    console.log(`Server URL: http://localhost:${PORT}`);
    console.log(`Health Check: http://localhost:${PORT}/health`);
    console.log(`Pinecone Index: ${PINECONE_INDEX_NAME}`);
    console.log(`${"=".repeat(60)}\n`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received. Shutting down gracefully...');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('SIGINT received. Shutting down gracefully...');
    process.exit(0);
});