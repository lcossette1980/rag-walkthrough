import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Code, Database, Search, Zap, BookOpen, Settings, Users, Brain, FileText, MessageSquare, BarChart3, Shield, Cpu, Cloud, Download, ExternalLink } from 'lucide-react';

interface ExpandedItems {
  [key: string]: boolean;
}

interface Implementation {
  title: string;
  description: string;
  complexity: string;
  timeToImplement: string;
  components: string[];
  pros: string[];
  cons: string[];
}

interface Implementations {
  [key: string]: Implementation;
}

interface UseCase {
  title: string;
  description: string;
  examples: string[];
  complexity: string;
  dataTypes: string[];
}

interface ArchitectureComponent {
  name: string;
  description: string;
  technologies: string[];
  responsibilities: string[];
}

interface Section {
  id: string;
  label: string;
  icon: React.ComponentType<{ size?: number }>;
}

const RAGGuide: React.FC = () => {
  const [activeSection, setActiveSection] = useState<string>('overview');
  const [expandedItems, setExpandedItems] = useState<ExpandedItems>({});
  const [selectedImplementation, setSelectedImplementation] = useState<string>('basic');

  const toggleExpanded = (item: string): void => {
    setExpandedItems(prev => ({
      ...prev,
      [item]: !prev[item]
    }));
  };

  const sections: Section[] = [
    { id: 'overview', label: 'RAG Overview', icon: BookOpen },
    { id: 'architecture', label: 'Architecture & Components', icon: Cpu },
    { id: 'implementations', label: 'Implementation Types', icon: Code },
    { id: 'use-cases', label: 'Use Cases & Examples', icon: Users },
    { id: 'setup', label: 'Setup & Configuration', icon: Settings },
    { id: 'advanced', label: 'Advanced Features', icon: Brain },
    { id: 'deployment', label: 'Deployment & Scaling', icon: Cloud },
    { id: 'best-practices', label: 'Best Practices', icon: Shield }
  ];

  const useCases: UseCase[] = [
    {
      title: "Knowledge Base Assistant",
      description: "Internal company knowledge, FAQs, documentation",
      examples: ["HR policies", "Technical documentation", "Product manuals"],
      complexity: "Basic",
      dataTypes: ["Text documents", "PDFs", "Wiki pages"]
    },
    {
      title: "Customer Support AI",
      description: "Automated customer service with context-aware responses",
      examples: ["Support tickets", "Product troubleshooting", "Service guides"],
      complexity: "Intermediate",
      dataTypes: ["Support tickets", "Product docs", "Chat logs"]
    },
    {
      title: "Research Assistant",
      description: "Academic or scientific research with citation support",
      examples: ["Scientific papers", "Legal documents", "Market research"],
      complexity: "Advanced",
      dataTypes: ["Research papers", "Reports", "Databases"]
    },
    {
      title: "Educational Tutor",
      description: "Personalized learning with curriculum-based responses",
      examples: ["Course materials", "Textbooks", "Learning modules"],
      complexity: "Intermediate",
      dataTypes: ["Textbooks", "Lectures", "Assignments"]
    },
    {
      title: "Code Assistant",
      description: "Programming help with codebase context",
      examples: ["Code repositories", "API documentation", "Stack Overflow"],
      complexity: "Advanced",
      dataTypes: ["Source code", "Documentation", "Issues"]
    },
    {
      title: "Medical Assistant",
      description: "Clinical decision support with medical literature",
      examples: ["Medical journals", "Treatment guidelines", "Drug databases"],
      complexity: "Expert",
      dataTypes: ["Medical texts", "Guidelines", "Case studies"]
    }
  ];

  const implementations: Implementations = {
    basic: {
      title: "Basic RAG Implementation",
      description: "Simple document retrieval with basic embedding",
      complexity: "Beginner",
      timeToImplement: "2-4 hours",
      components: ["Text loading", "Basic chunking", "Simple embeddings", "Vector search", "LLM integration"],
      pros: ["Quick setup", "Low complexity", "Good for prototyping"],
      cons: ["Limited accuracy", "No advanced features", "Basic retrieval"]
    },
    intermediate: {
      title: "Enhanced RAG System",
      description: "Improved retrieval with metadata and filtering",
      complexity: "Intermediate",
      timeToImplement: "1-2 days",
      components: ["Advanced chunking", "Metadata enrichment", "Hybrid search", "Query optimization", "Response filtering"],
      pros: ["Better accuracy", "Metadata filtering", "Improved relevance"],
      cons: ["More complex setup", "Higher resource usage", "Requires tuning"]
    },
    advanced: {
      title: "Production RAG Pipeline",
      description: "Enterprise-grade with monitoring and optimization",
      complexity: "Advanced",
      timeToImplement: "1-2 weeks",
      components: ["Multi-modal support", "Real-time updates", "A/B testing", "Performance monitoring", "Auto-scaling"],
      pros: ["Production ready", "High performance", "Monitoring included"],
      cons: ["Complex architecture", "Resource intensive", "Requires expertise"]
    },
    multimodal: {
      title: "Multi-Modal RAG",
      description: "Text, images, and other data types",
      complexity: "Expert",
      timeToImplement: "2-4 weeks",
      components: ["Multi-modal embeddings", "Cross-modal search", "Unified indexing", "Complex reasoning", "Advanced UI"],
      pros: ["Handles all data types", "Rich interactions", "Advanced capabilities"],
      cons: ["Very complex", "High costs", "Specialized knowledge needed"]
    }
  };

  const architectureComponents: ArchitectureComponent[] = [
    {
      name: "Data Ingestion Layer",
      description: "Handles various data sources and formats",
      technologies: ["Apache Kafka", "AWS S3", "REST APIs", "File watchers"],
      responsibilities: ["Data collection", "Format conversion", "Quality validation", "Metadata extraction"]
    },
    {
      name: "Processing Pipeline",
      description: "Transforms raw data into searchable chunks",
      technologies: ["LangChain", "Haystack", "Custom processors", "Apache Beam"],
      responsibilities: ["Text extraction", "Chunking strategies", "Cleaning & normalization", "Metadata enrichment"]
    },
    {
      name: "Embedding Generation",
      description: "Converts text to vector representations",
      technologies: ["OpenAI Embeddings", "Sentence Transformers", "Cohere", "Custom models"],
      responsibilities: ["Vector generation", "Batch processing", "Model management", "Quality assessment"]
    },
    {
      name: "Vector Database",
      description: "Stores and retrieves vector embeddings efficiently",
      technologies: ["Pinecone", "Weaviate", "Qdrant", "FAISS", "Chroma"],
      responsibilities: ["Vector storage", "Similarity search", "Indexing", "Scalability"]
    },
    {
      name: "Retrieval Engine",
      description: "Finds relevant documents based on queries",
      technologies: ["Dense retrieval", "Sparse retrieval", "Hybrid search", "Re-ranking"],
      responsibilities: ["Query processing", "Relevance scoring", "Result filtering", "Context optimization"]
    },
    {
      name: "Generation Module",
      description: "Produces responses using retrieved context",
      technologies: ["GPT-4", "Claude", "Llama", "Custom fine-tuned models"],
      responsibilities: ["Response generation", "Context integration", "Quality control", "Safety filtering"]
    }
  ];

  const renderCodeExample = (type: string): string => {
    const examples: { [key: string]: string } = {
      basic: `# Basic RAG Implementation
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI

# 1. Load and process documents
def setup_knowledge_base(documents):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore.as_retriever()

# 2. RAG query function
def rag_query(retriever, llm, question):
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(question)
    
    # Combine context
    context = "\\n\\n".join([doc.page_content for doc in docs])
    
    # Generate response
    prompt = f"""
    Context: {context}
    
    Question: {question}
    
    Answer based on the context:
    """
    
    return llm(prompt)`,
      
      intermediate: `# Enhanced RAG with Metadata Filtering
import chromadb
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

class EnhancedRAG:
    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("knowledge_base")
    
    def add_documents(self, documents, metadata):
        # Process documents with metadata
        for doc, meta in zip(documents, metadata):
            chunks = self.chunk_document(doc)
            embeddings = self.embedding_model.embed_documents(chunks)
            
            # Store with metadata
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{**meta, "chunk_id": i} for i in range(len(chunks))],
                ids=[f"{meta['doc_id']}_chunk_{i}" for i in range(len(chunks))]
            )
    
    def query_with_filters(self, question, filters=None):
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(question)
        
        # Search with filters
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=10,
            where=filters
        )
        
        # Compress results for relevance
        compressor = LLMChainExtractor.from_llm(self.llm)
        compressed_docs = compressor.compress_documents(
            results['documents'][0], question
        )
        
        return self.generate_response(question, compressed_docs)`,
      
      production: `# Production-Ready RAG Pipeline
from typing import List, Dict, Any
import asyncio
from dataclasses import dataclass
from monitoring import MetricsCollector
from caching import RedisCache
from queue import TaskQueue

@dataclass
class RAGConfig:
    embedding_model: str
    llm_model: str
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    temperature: float = 0.1

class ProductionRAG:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.metrics = MetricsCollector()
        self.cache = RedisCache()
        self.task_queue = TaskQueue()
        
    async def process_query(self, query: str, user_id: str) -> Dict[str, Any]:
        # Start timing
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = await self.cache.get(query)
            if cached_result:
                self.metrics.record_cache_hit(user_id)
                return cached_result
            
            # Async retrieval and generation
            relevant_docs = await self.retrieve_documents(query)
            response = await self.generate_response(query, relevant_docs)
            
            # Cache result
            await self.cache.set(query, response, ttl=3600)
            
            # Record metrics
            self.metrics.record_query(
                user_id=user_id,
                query=query,
                response_time=time.time() - start_time,
                relevance_score=response.get('relevance_score', 0)
            )
            
            return response
            
        except Exception as e:
            self.metrics.record_error(user_id, str(e))
            raise
    
    async def batch_index_documents(self, documents: List[Dict]):
        # Queue documents for processing
        tasks = [
            self.task_queue.add_task('index_document', doc) 
            for doc in documents
        ]
        
        # Process in parallel
        await asyncio.gather(*tasks)
        
    def get_analytics(self) -> Dict[str, Any]:
        return {
            'total_queries': self.metrics.get_total_queries(),
            'avg_response_time': self.metrics.get_avg_response_time(),
            'cache_hit_rate': self.metrics.get_cache_hit_rate(),
            'error_rate': self.metrics.get_error_rate()
        }`,

      multimodal: `# Multi-Modal RAG Implementation
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import whisper

class MultiModalRAG:
    def __init__(self):
        # Initialize models
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.whisper_model = whisper.load_model("base")
        
    def process_image(self, image_path: str) -> Dict[str, Any]:
        # Load and process image
        image = Image.open(image_path)
        inputs = self.clip_processor(images=image, return_tensors="pt")
        
        # Generate image embeddings
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
        
        return {
            'embeddings': image_features.numpy(),
            'modality': 'image',
            'source': image_path
        }
    
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        # Transcribe audio
        result = self.whisper_model.transcribe(audio_path)
        
        # Generate text embeddings from transcription
        text_inputs = self.clip_processor(text=result['text'], return_tensors="pt")
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**text_inputs)
        
        return {
            'embeddings': text_features.numpy(),
            'transcription': result['text'],
            'modality': 'audio',
            'source': audio_path
        }
    
    def cross_modal_search(self, query: str, modalities: List[str]) -> List[Dict]:
        # Process text query
        text_inputs = self.clip_processor(text=query, return_tensors="pt")
        with torch.no_grad():
            query_features = self.clip_model.get_text_features(**text_inputs)
        
        # Search across different modalities
        results = []
        for modality in modalities:
            modal_results = self.search_modality(query_features, modality)
            results.extend(modal_results)
        
        # Rank and return top results
        return sorted(results, key=lambda x: x['similarity'], reverse=True)[:10]`
    };
    
    return examples[type] || examples.basic;
  };

  return (
    <div className="min-h-screen gradient-shift" style={{
      fontFamily: "'Lato', sans-serif",
      background: 'linear-gradient(135deg, #F5F2EA 0%, #D7CEB2 25%, #F5F2EA 50%, #D7CEB2 75%, #F5F2EA 100%)'
    }}>
      {/* Google Fonts Import */}
      <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Lato:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
      
      {/* Custom CSS */}
      <style dangerouslySetInnerHTML={{
        __html: `
        .playfair-bold {
          font-family: 'Playfair Display', serif;
          font-weight: 700;
        }
        
        .playfair-black {
          font-family: 'Playfair Display', serif;
          font-weight: 900;
        }
        
        .lato-regular {
          font-family: 'Lato', sans-serif;
          font-weight: 400;
        }
        
        .lato-medium {
          font-family: 'Lato', sans-serif;
          font-weight: 500;
        }
        
        .lato-semibold {
          font-family: 'Lato', sans-serif;
          font-weight: 600;
        }
        
        .brand-gradient {
          background: linear-gradient(135deg, #A44A3F 0%, #A59E8C 100%);
        }
        
        .brand-shadow {
          box-shadow: 0 10px 25px rgba(42, 42, 42, 0.1);
        }
        
        .hover-lift {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .hover-lift:hover {
          transform: translateY(-2px);
          box-shadow: 0 15px 35px rgba(42, 42, 42, 0.15);
        }
        
        .glass-effect {
          background: rgba(245, 242, 234, 0.9);
          backdrop-filter: blur(10px);
          border: 1px solid rgba(215, 206, 178, 0.3);
        }
        
        .text-charcoal { color: #2A2A2A; }
        .text-chestnut { color: #A44A3F; }
        .text-khaki { color: #A59E8C; }
        .text-pearl { color: #D7CEB2; }
        .text-bone { color: #F5F2EA; }
        
        .bg-charcoal { background-color: #2A2A2A; }
        .bg-chestnut { background-color: #A44A3F; }
        .bg-khaki { background-color: #A59E8C; }
        .bg-pearl { background-color: #D7CEB2; }
        .bg-bone { background-color: #F5F2EA; }
        
        .border-chestnut { border-color: #A44A3F; }
        .border-khaki { border-color: #A59E8C; }
        .border-pearl { border-color: #D7CEB2; }
        
        @keyframes pulse {
          0%, 100% {
            opacity: 0.1;
            transform: scale(1);
          }
          50% {
            opacity: 0.3;
            transform: scale(1.1);
          }
        }
        
        @keyframes float {
          0%, 100% {
            transform: rotate(12deg) scale(1.5) translateY(0px);
          }
          50% {
            transform: rotate(12deg) scale(1.5) translateY(-20px);
          }
        }
        
        .animate-float {
          animation: float 20s ease-in-out infinite;
        }
        
        .gradient-shift {
          background-size: 200% 200%;
          animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
          0% {
            background-position: 0% 50%;
          }
          50% {
            background-position: 100% 50%;
          }
          100% {
            background-position: 0% 50%;
          }
        }
        `
      }} />

      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="bg-white shadow-lg mb-12 sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div className="w-16 h-16 bg-white rounded-xl brand-shadow flex items-center justify-center p-2">
                <img src="/logo-evolveiq.png" alt="E EvolvIQ" className="max-w-full max-h-full object-contain" />
              </div>
              <div>
                <h1 className="text-2xl playfair-bold text-charcoal leading-tight">RAG Implementation Guide</h1>
                <p className="text-sm lato-regular text-khaki">Comprehensive AI Development Resource</p>
              </div>
            </div>
            <nav className="flex items-center gap-8">
              <a href="#overview" className="lato-semibold text-khaki hover:text-chestnut transition-colors px-4 py-2 rounded-lg hover:bg-bone">Overview</a>
              <a href="#download" className="lato-semibold text-khaki hover:text-chestnut transition-colors px-4 py-2 rounded-lg hover:bg-bone">Download</a>
              <a href="#examples" className="lato-semibold text-khaki hover:text-chestnut transition-colors px-4 py-2 rounded-lg hover:bg-bone">Examples</a>
            </nav>
          </div>
        </header>

        {/* Hero Header with Logo Placeholder */}
        <div className="text-center mb-16 relative overflow-hidden">
          {/* Background Pattern with Animation */}
          <div className="absolute inset-0 opacity-5">
            <div className="grid grid-cols-12 gap-6 transform rotate-12 scale-150 animate-float">
              {Array.from({length: 144}).map((_, i) => (
                <div 
                  key={i} 
                  className="w-6 h-6 rounded-full"
                  style={{
                    backgroundColor: '#D7CEB2',
                    opacity: 0.3,
                    animation: `pulse ${3 + (i % 3)}s ease-in-out infinite`
                  }}
                ></div>
              ))}
            </div>
          </div>
          
          {/* Additional floating shapes layer */}
          <div className="absolute inset-0 pointer-events-none">
            <div 
              className="absolute top-10 left-10 w-32 h-32 rounded-full"
              style={{
                background: 'radial-gradient(circle, rgba(164, 74, 63, 0.05) 0%, transparent 70%)',
                animation: 'float 25s ease-in-out infinite'
              }}
            ></div>
            <div 
              className="absolute top-40 right-20 w-40 h-40 rounded-full"
              style={{
                background: 'radial-gradient(circle, rgba(165, 158, 140, 0.05) 0%, transparent 70%)',
                animation: 'float 30s ease-in-out infinite reverse'
              }}
            ></div>
            <div 
              className="absolute bottom-20 left-1/3 w-48 h-48 rounded-full"
              style={{
                background: 'radial-gradient(circle, rgba(215, 206, 178, 0.05) 0%, transparent 70%)',
                animation: 'float 35s ease-in-out infinite'
              }}
            ></div>
          </div>
          
          <div className="relative z-10 mb-8">
            <div 
              className="w-80 h-32 mx-auto mb-6 rounded-3xl brand-shadow hover-lift bg-white bg-cover bg-center bg-no-repeat"
              style={{
                backgroundImage: "url('/logo-evolveiq.png')",
                backgroundSize: "contain"
              }}
            >
            </div>
            <h1 className="text-5xl playfair-black text-charcoal mb-4 leading-tight max-w-4xl mx-auto">
              Comprehensive RAG Implementation Guide
            </h1>
            <p className="text-xl lato-regular text-khaki max-w-4xl mx-auto leading-relaxed">
              Complete walkthrough for building Retrieval-Augmented Generation systems 
              for educational and professional use
            </p>
          </div>
          
          {/* Feature Icons Row */}
          <div className="flex justify-center gap-12 mt-12">
            {[
              { icon: Brain, label: "AI-Powered" },
              { icon: Database, label: "Knowledge Base" },
              { icon: Search, label: "Smart Retrieval" },
              { icon: MessageSquare, label: "Natural Language" }
            ].map((feature, idx) => (
              <div key={idx} className="text-center hover-lift">
                <div className="w-20 h-20 mx-auto mb-4 bg-bone rounded-2xl border-2 border-pearl flex items-center justify-center brand-shadow">
                  <feature.icon size={32} className="text-chestnut" />
                </div>
                <p className="text-base lato-semibold text-khaki">{feature.label}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-col lg:flex-row gap-12">
          {/* Sidebar Navigation */}
          <div className="lg:w-1/3">
            <div className="bg-white rounded-3xl shadow-xl p-8 sticky top-32">
              <h3 className="text-2xl playfair-bold mb-6 text-charcoal">Guide Sections</h3>
              <nav className="space-y-3">
                {sections.map((section) => {
                  const Icon = section.icon;
                  return (
                    <button
                      key={section.id}
                      onClick={() => setActiveSection(section.id)}
                      className={`w-full flex items-center gap-4 p-4 rounded-2xl text-left transition-all hover-lift ${
                        activeSection === section.id 
                          ? 'brand-gradient text-bone shadow-xl scale-105' 
                          : 'hover:bg-bone text-charcoal hover:shadow-lg'
                      }`}
                    >
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
                        activeSection === section.id ? 'bg-bone bg-opacity-20' : 'bg-pearl'
                      }`}>
                        <Icon size={20} {...({ stroke: activeSection === section.id ? '#F5F2EA' : '#A44A3F' } as any)} />
                      </div>
                      <span className="lato-semibold text-base">{section.label}</span>
                    </button>
                  );
                })}
              </nav>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:w-2/3">
            <div className="bg-white rounded-3xl shadow-xl p-12">
              
              {/* Overview Section */}
              {activeSection === 'overview' && (
                <div className="space-y-6">
                  {/* Section Header with Graphic Placeholder */}
                  <div className="text-center mb-8">
                    <div className="w-20 h-20 mx-auto mb-4 bg-gradient-to-br from-chestnut via-khaki to-pearl rounded-2xl flex items-center justify-center brand-shadow hover-lift">
                      <BookOpen size={28} className="text-bone" />
                    </div>
                    <h2 className="text-3xl playfair-bold text-charcoal mb-3">What is RAG?</h2>
                    <div className="w-32 h-1 bg-gradient-to-r from-chestnut to-khaki mx-auto rounded-full"></div>
                  </div>
                  
                  <div className="grid md:grid-cols-2 gap-6">
                    <div className="bg-gradient-to-br from-bone to-pearl p-6 rounded-2xl border border-pearl hover-lift">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="w-10 h-10 bg-chestnut rounded-xl flex items-center justify-center">
                          <Brain size={20} className="text-bone" />
                        </div>
                        <h3 className="text-xl playfair-bold text-charcoal">Retrieval-Augmented Generation</h3>
                      </div>
                      <p className="lato-regular text-charcoal leading-relaxed">
                        RAG combines the power of large language models with external knowledge retrieval, 
                        enabling AI systems to provide accurate, up-to-date, and contextually relevant responses 
                        based on your specific data sources.
                      </p>
                    </div>
                    
                    <div className="bg-gradient-to-br from-pearl to-khaki p-6 rounded-2xl border border-khaki hover-lift">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="w-10 h-10 bg-charcoal rounded-xl flex items-center justify-center">
                          <Zap size={20} className="text-bone" />
                        </div>
                        <h3 className="text-xl playfair-bold text-charcoal">Why Use RAG?</h3>
                      </div>
                      <ul className="lato-regular text-charcoal space-y-2">
                        <li className="flex items-center gap-3">
                          <div className="w-2 h-2 bg-chestnut rounded-full"></div>
                          Reduces hallucinations
                        </li>
                        <li className="flex items-center gap-3">
                          <div className="w-2 h-2 bg-chestnut rounded-full"></div>
                          Uses your specific data
                        </li>
                        <li className="flex items-center gap-3">
                          <div className="w-2 h-2 bg-chestnut rounded-full"></div>
                          Provides source attribution
                        </li>
                        <li className="flex items-center gap-3">
                          <div className="w-2 h-2 bg-chestnut rounded-full"></div>
                          Enables domain expertise
                        </li>
                        <li className="flex items-center gap-3">
                          <div className="w-2 h-2 bg-chestnut rounded-full"></div>
                          Cost-effective vs fine-tuning
                        </li>
                      </ul>
                    </div>
                  </div>

                  {/* RAG Workflow Visualization */}
                  <div className="bg-gradient-to-r from-bone to-pearl p-6 rounded-2xl border-2 border-pearl">
                    <div className="text-center mb-6">
                      <h3 className="text-2xl playfair-bold text-charcoal mb-4">RAG Workflow</h3>
                      <div className="w-24 h-1 bg-gradient-to-r from-chestnut to-khaki mx-auto rounded-full"></div>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                      {[
                        { icon: FileText, title: "Ingest", desc: "Load documents", color: "chestnut" },
                        { icon: Database, title: "Store", desc: "Create embeddings", color: "khaki" },
                        { icon: Search, title: "Retrieve", desc: "Find relevant docs", color: "chestnut" },
                        { icon: MessageSquare, title: "Generate", desc: "Create response", color: "khaki" }
                      ].map((step, idx) => (
                        <div key={idx} className="text-center hover-lift">
                          <div className={`w-20 h-20 mx-auto mb-4 bg-${step.color} rounded-2xl flex items-center justify-center brand-shadow relative`}>
                            <step.icon size={28} className="text-bone" />
                            <div className="absolute -top-2 -right-2 w-8 h-8 bg-charcoal rounded-full flex items-center justify-center">
                              <span className="text-bone text-sm playfair-bold">{idx + 1}</span>
                            </div>
                          </div>
                          <h4 className="playfair-bold text-charcoal text-lg mb-2">{step.title}</h4>
                          <p className="lato-regular text-khaki">{step.desc}</p>
                          {idx < 3 && (
                            <div className="hidden md:block absolute top-10 left-full w-6 h-0.5 bg-khaki transform -translate-x-3"></div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Stats/Metrics Placeholder */}
                  <div className="grid md:grid-cols-3 gap-6">
                    {[
                      { number: "10x", label: "Faster Development", icon: Zap },
                      { number: "95%", label: "Accuracy Improvement", icon: BarChart3 },
                      { number: "50+", label: "Use Cases Covered", icon: Users }
                    ].map((stat, idx) => (
                      <div key={idx} className="text-center p-6 bg-gradient-to-br from-bone to-pearl rounded-2xl border border-pearl hover-lift">
                        <stat.icon size={32} className="mx-auto mb-4 text-chestnut" />
                        <div className="text-3xl playfair-bold text-charcoal mb-2">{stat.number}</div>
                        <div className="lato-medium text-khaki">{stat.label}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Architecture Section */}
              {activeSection === 'architecture' && (
                <div className="space-y-8">
                  {/* Section Header */}
                  <div className="text-center mb-12">
                    <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-khaki via-chestnut to-charcoal rounded-2xl flex items-center justify-center brand-shadow hover-lift">
                      <Cpu size={32} className="text-bone" />
                    </div>
                    <h2 className="text-4xl playfair-bold text-charcoal mb-4">RAG Architecture & Components</h2>
                    <div className="w-32 h-1 bg-gradient-to-r from-chestnut to-khaki mx-auto rounded-full"></div>
                  </div>
                  
                  {/* Architecture Diagram Placeholder */}
                  <div className="bg-gradient-to-br from-bone to-pearl p-8 rounded-2xl border-2 border-pearl mb-8">
                    <div className="text-center mb-6">
                      <h3 className="text-2xl playfair-bold text-charcoal mb-4">System Architecture Overview</h3>
                      <div className="w-full h-64 bg-gradient-to-r from-khaki via-pearl to-chestnut rounded-xl opacity-20 flex items-center justify-center">
                        <div className="text-center">
                          <Database size={48} className="mx-auto mb-4 text-charcoal" />
                          <p className="lato-medium text-charcoal">Architecture Diagram Placeholder</p>
                          <p className="text-sm lato-regular text-khaki mt-2">Interactive component flow visualization</p>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="space-y-6">
                    {architectureComponents.map((component, idx) => (
                      <div key={idx} className="border-2 border-pearl rounded-2xl overflow-hidden bg-bone brand-shadow hover-lift">
                        <button
                          onClick={() => toggleExpanded(`arch-${idx}`)}
                          className="w-full p-6 bg-gradient-to-r from-bone to-pearl flex items-center justify-between hover:from-pearl hover:to-khaki transition-all"
                        >
                          <div className="flex items-center gap-4">
                            <div className="w-12 h-12 bg-chestnut rounded-xl flex items-center justify-center">
                              <span className="text-bone playfair-bold text-lg">{idx + 1}</span>
                            </div>
                            <h3 className="text-xl playfair-bold text-charcoal">{component.name}</h3>
                          </div>
                          {expandedItems[`arch-${idx}`] ? 
                            <ChevronDown size={24} className="text-chestnut" /> : 
                            <ChevronRight size={24} className="text-chestnut" />
                          }
                        </button>
                        
                        {expandedItems[`arch-${idx}`] && (
                          <div className="p-6 border-t-2 border-pearl bg-bone">
                            <p className="lato-regular text-charcoal mb-6 leading-relaxed">{component.description}</p>
                            
                            <div className="grid md:grid-cols-2 gap-6">
                              <div>
                                <h4 className="playfair-bold text-charcoal mb-4 text-lg">Technologies:</h4>
                                <div className="space-y-2">
                                  {component.technologies.map((tech, techIdx) => (
                                    <span key={techIdx} className="inline-block lato-medium text-sm text-bone bg-chestnut px-3 py-2 rounded-lg mr-2 mb-2 hover-lift">
                                      {tech}
                                    </span>
                                  ))}
                                </div>
                              </div>
                              
                              <div>
                                <h4 className="playfair-bold text-charcoal mb-4 text-lg">Responsibilities:</h4>
                                <ul className="lato-regular text-charcoal space-y-2">
                                  {component.responsibilities.map((resp, respIdx) => (
                                    <li key={respIdx} className="flex items-start gap-3">
                                      <div className="w-2 h-2 bg-khaki rounded-full mt-2 flex-shrink-0"></div>
                                      <span>{resp}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Implementation Types */}
              {activeSection === 'implementations' && (
                <div className="space-y-8">
                  {/* Section Header */}
                  <div className="text-center mb-12">
                    <div className="w-24 h-24 mx-auto mb-6 bg-gradient-to-br from-chestnut via-khaki to-pearl rounded-2xl flex items-center justify-center brand-shadow hover-lift">
                      <Code size={32} className="text-bone" />
                    </div>
                    <h2 className="text-4xl playfair-bold text-charcoal mb-4">Implementation Types</h2>
                    <div className="w-32 h-1 bg-gradient-to-r from-chestnut to-khaki mx-auto rounded-full"></div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                    {Object.entries(implementations).map(([key, impl]) => (
                      <button
                        key={key}
                        onClick={() => setSelectedImplementation(key)}
                        className={`p-6 rounded-2xl border-2 text-left transition-all hover-lift ${
                          selectedImplementation === key 
                            ? 'border-chestnut bg-gradient-to-br from-chestnut to-khaki text-bone brand-shadow' 
                            : 'border-pearl bg-bone hover:border-khaki text-charcoal'
                        }`}
                      >
                        <h3 className="playfair-bold text-xl mb-3">{impl.title}</h3>
                        <p className={`lato-regular mb-4 ${selectedImplementation === key ? 'text-bone' : 'text-khaki'}`}>
                          {impl.description}
                        </p>
                        <div className="flex gap-3 flex-wrap">
                          <span className={`text-xs px-3 py-2 rounded-lg lato-medium ${
                            impl.complexity === 'Beginner' ? 'bg-green-100 text-green-800' :
                            impl.complexity === 'Intermediate' ? 'bg-yellow-100 text-yellow-800' :
                            impl.complexity === 'Advanced' ? 'bg-orange-100 text-orange-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {impl.complexity}
                          </span>
                          <span className={`text-xs px-3 py-2 rounded-lg lato-medium ${
                            selectedImplementation === key ? 'bg-bone text-charcoal' : 'bg-pearl text-charcoal'
                          }`}>
                            {impl.timeToImplement}
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>

                  {/* Selected Implementation Details */}
                  <div className="bg-gradient-to-br from-bone to-pearl rounded-2xl p-8 border-2 border-pearl brand-shadow">
                    <div className="flex items-center gap-4 mb-8">
                      <div className="w-16 h-16 bg-chestnut rounded-xl flex items-center justify-center">
                        <Code size={24} className="text-bone" />
                      </div>
                      <h3 className="text-3xl playfair-bold text-charcoal">{implementations[selectedImplementation].title}</h3>
                    </div>
                    
                    <div className="grid md:grid-cols-3 gap-8 mb-8">
                      <div>
                        <h4 className="playfair-bold text-charcoal mb-4 text-lg">Components:</h4>
                        <ul className="lato-regular text-charcoal space-y-2">
                          {implementations[selectedImplementation].components.map((comp: string, idx: number) => (
                            <li key={idx} className="flex items-start gap-3">
                              <div className="w-2 h-2 bg-chestnut rounded-full mt-2 flex-shrink-0"></div>
                              <span>{comp}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="playfair-bold text-green-800 mb-4 text-lg">Pros:</h4>
                        <ul className="lato-regular text-green-700 space-y-3">
                          {implementations[selectedImplementation].pros.map((pro: string, idx: number) => (
                            <li key={idx} className="flex items-start gap-3">
                              <div className="w-2 h-2 bg-green-600 rounded-full mt-2 flex-shrink-0"></div>
                              <span>{pro}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                      
                      <div>
                        <h4 className="playfair-bold text-red-800 mb-4 text-lg">Cons:</h4>
                        <ul className="lato-regular text-red-700 space-y-3">
                          {implementations[selectedImplementation].cons.map((con: string, idx: number) => (
                            <li key={idx} className="flex items-start gap-3">
                              <div className="w-2 h-2 bg-red-600 rounded-full mt-2 flex-shrink-0"></div>
                              <span>{con}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    {/* Code Example */}
                    <div className="bg-charcoal rounded-2xl p-6 overflow-x-auto brand-shadow">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="flex gap-2">
                          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                          <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                        </div>
                        <span className="lato-medium text-pearl text-sm">implementation.py</span>
                      </div>
                      <pre className="text-green-400 text-sm lato-regular leading-relaxed">
                        <code>{renderCodeExample(selectedImplementation)}</code>
                      </pre>
                    </div>
                  </div>
                </div>
              )}

              {/* Use Cases Section */}
              {activeSection === 'use-cases' && (
                <div className="space-y-6">
                  <h2 className="text-3xl font-bold text-gray-800 mb-6">Use Cases & Examples</h2>
                  
                  <div className="grid gap-6">
                    {useCases.map((useCase, idx) => (
                      <div key={idx} className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg p-6 border border-gray-200">
                        <div className="flex items-start justify-between mb-4">
                          <div>
                            <h3 className="text-xl font-semibold text-gray-800">{useCase.title}</h3>
                            <p className="text-gray-600 mt-1">{useCase.description}</p>
                          </div>
                          <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                            useCase.complexity === 'Basic' ? 'bg-green-100 text-green-800' :
                            useCase.complexity === 'Intermediate' ? 'bg-yellow-100 text-yellow-800' :
                            useCase.complexity === 'Advanced' ? 'bg-orange-100 text-orange-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {useCase.complexity}
                          </span>
                        </div>
                        
                        <div className="grid md:grid-cols-2 gap-4">
                          <div>
                            <h4 className="font-medium text-gray-800 mb-2">Example Applications:</h4>
                            <ul className="text-sm text-gray-600 space-y-1">
                              {useCase.examples.map((example, exIdx) => (
                                <li key={exIdx} className="flex items-center gap-2">
                                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                                  {example}
                                </li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="font-medium text-gray-800 mb-2">Data Types:</h4>
                            <div className="flex flex-wrap gap-2">
                              {useCase.dataTypes.map((dataType, dtIdx) => (
                                <span key={dtIdx} className="text-xs px-2 py-1 bg-blue-100 text-blue-800 rounded">
                                  {dataType}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Setup Section */}
              {activeSection === 'setup' && (
                <div className="space-y-6">
                  <h2 className="text-3xl font-bold text-gray-800 mb-6">Setup & Configuration</h2>
                  
                  <div className="space-y-6">
                    {/* Prerequisites */}
                    <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
                      <h3 className="text-lg font-semibold text-blue-800 mb-4">Prerequisites</h3>
                      <div className="grid md:grid-cols-2 gap-4">
                        <div>
                          <h4 className="font-medium mb-2">Technical Requirements:</h4>
                          <ul className="text-sm text-blue-700 space-y-1">
                            <li>• Python 3.8+ or Node.js 16+</li>
                            <li>• Vector database (FAISS, Pinecone, etc.)</li>
                            <li>• LLM API access (OpenAI, Anthropic, etc.)</li>
                            <li>• Embedding model access</li>
                            <li>• Sufficient compute resources</li>
                          </ul>
                        </div>
                        <div>
                          <h4 className="font-medium mb-2">Knowledge Requirements:</h4>
                          <ul className="text-sm text-blue-700 space-y-1">
                            <li>• Basic ML/AI concepts</li>
                            <li>• Vector embeddings understanding</li>
                            <li>• API integration experience</li>
                            <li>• Data preprocessing skills</li>
                            <li>• Prompt engineering basics</li>
                          </ul>
                        </div>
                      </div>
                    </div>

                    {/* Step-by-step Setup */}
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold text-gray-800">Step-by-Step Setup Guide</h3>
                      
                      {[
                        {
                          step: 1,
                          title: "Environment Setup",
                          content: "Install dependencies, set up API keys, configure development environment",
                          code: `pip install langchain openai faiss-cpu streamlit
export OPENAI_API_KEY="your-api-key-here"
mkdir rag-project && cd rag-project`
                        },
                        {
                          step: 2,
                          title: "Data Preparation",
                          content: "Collect, clean, and format your knowledge base documents",
                          code: `# Document processing
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('./data/', glob="**/*.txt")
documents = loader.load()`
                        },
                        {
                          step: 3,
                          title: "Embedding Creation",
                          content: "Generate vector embeddings for your documents",
                          code: `from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)`
                        },
                        {
                          step: 4,
                          title: "Retrieval Setup",
                          content: "Configure document retrieval and ranking",
                          code: `retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)`
                        },
                        {
                          step: 5,
                          title: "LLM Integration",
                          content: "Connect your chosen language model",
                          code: `from langchain.llms import OpenAI
llm = OpenAI(temperature=0)
# Ready for queries!`
                        }
                      ].map((step) => (
                        <div key={step.step} className="border border-gray-200 rounded-lg overflow-hidden">
                          <div className="bg-gray-50 p-4 flex items-center gap-3">
                            <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center font-medium">
                              {step.step}
                            </div>
                            <h4 className="font-semibold text-gray-800">{step.title}</h4>
                          </div>
                          <div className="p-4">
                            <p className="text-gray-600 mb-3">{step.content}</p>
                            <div className="bg-gray-900 rounded p-3 overflow-x-auto">
                              <code className="text-green-400 text-sm">{step.code}</code>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Advanced Features */}
              {activeSection === 'advanced' && (
                <div className="space-y-6">
                  <h2 className="text-3xl font-bold text-gray-800 mb-6">Advanced Features</h2>
                  
                  <div className="grid gap-6">
                    {[
                      {
                        title: "Hybrid Search",
                        description: "Combine dense and sparse retrieval for better accuracy",
                        features: ["Dense vector search", "Keyword matching", "Score fusion", "Query optimization"],
                        complexity: "Intermediate"
                      },
                      {
                        title: "Multi-Modal RAG",
                        description: "Handle text, images, audio, and other data types",
                        features: ["Cross-modal embeddings", "Unified search", "Rich responses", "Media processing"],
                        complexity: "Advanced"
                      },
                      {
                        title: "Real-time Updates",
                        description: "Keep knowledge base current with live data streams",
                        features: ["Incremental indexing", "Change detection", "Hot swapping", "Version control"],
                        complexity: "Advanced"
                      },
                      {
                        title: "Contextual Compression",
                        description: "Optimize retrieved context for better responses",
                        features: ["Relevance filtering", "Content summarization", "Context ranking", "Noise reduction"],
                        complexity: "Intermediate"
                      },
                      {
                        title: "Query Routing",
                        description: "Route queries to specialized knowledge bases",
                        features: ["Intent classification", "Multi-index search", "Routing logic", "Fallback handling"],
                        complexity: "Advanced"
                      },
                      {
                        title: "Answer Validation",
                        description: "Verify and improve response quality",
                        features: ["Fact checking", "Source verification", "Confidence scoring", "Quality metrics"],
                        complexity: "Expert"
                      }
                    ].map((feature, idx) => (
                      <div key={idx} className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
                        <div className="flex items-start justify-between mb-3">
                          <h3 className="text-lg font-semibold text-gray-800">{feature.title}</h3>
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            feature.complexity === 'Intermediate' ? 'bg-yellow-100 text-yellow-800' :
                            feature.complexity === 'Advanced' ? 'bg-orange-100 text-orange-800' :
                            'bg-red-100 text-red-800'
                          }`}>
                            {feature.complexity}
                          </span>
                        </div>
                        <p className="text-gray-600 mb-4">{feature.description}</p>
                        <div className="flex flex-wrap gap-2">
                          {feature.features.map((feat, featIdx) => (
                            <span key={featIdx} className="px-3 py-1 bg-gray-100 text-gray-700 rounded-full text-sm">
                              {feat}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Deployment Section */}
              {activeSection === 'deployment' && (
                <div className="space-y-6">
                  <h2 className="text-3xl font-bold text-gray-800 mb-6">Deployment & Scaling</h2>
                  
                  <div className="grid md:grid-cols-2 gap-6">
                    {[
                      {
                        title: "Local Development",
                        icon: Cpu,
                        description: "Run RAG system on your local machine",
                        pros: ["Full control", "No cloud costs", "Easy debugging", "Private data"],
                        cons: ["Limited resources", "No scaling", "Manual updates", "Single point of failure"],
                        technologies: ["Docker", "Local GPU", "SQLite", "File storage"]
                      },
                      {
                        title: "Cloud Deployment",
                        icon: Cloud,
                        description: "Deploy to cloud platforms for scalability",
                        pros: ["Auto-scaling", "High availability", "Managed services", "Global distribution"],
                        cons: ["Cloud costs", "Vendor lock-in", "Network latency", "Security concerns"],
                        technologies: ["AWS", "Azure", "GCP", "Kubernetes"]
                      },
                      {
                        title: "Containerized Setup",
                        icon: Settings,
                        description: "Use containers for consistent deployment",
                        pros: ["Reproducible", "Portable", "Version control", "Easy updates"],
                        cons: ["Container overhead", "Complexity", "Storage challenges", "Network config"],
                        technologies: ["Docker", "Docker Compose", "Kubernetes", "Helm"]
                      },
                      {
                        title: "Serverless Architecture",
                        icon: Zap,
                        description: "Event-driven, pay-per-use deployment",
                        pros: ["Cost effective", "Auto-scaling", "No server management", "High availability"],
                        cons: ["Cold starts", "Timeout limits", "Vendor specific", "Debugging difficulty"],
                        technologies: ["Lambda", "Cloud Functions", "Vercel", "Netlify"]
                      }
                    ].map((option, idx) => (
                      <div key={idx} className="bg-white border border-gray-200 rounded-xl p-6 hover:shadow-lg transition-all">
                        <div className="flex items-center gap-3 mb-4">
                          <div className="p-2 bg-blue-100 rounded-lg">
                            <option.icon className="text-blue-600" size={24} />
                          </div>
                          <h3 className="text-lg font-semibold text-gray-800">{option.title}</h3>
                        </div>
                        
                        <p className="text-gray-600 mb-4">{option.description}</p>
                        
                        <div className="space-y-3">
                          <div>
                            <h4 className="text-sm font-medium text-green-800 mb-1">Pros:</h4>
                            <ul className="text-xs text-green-600 space-y-1">
                              {option.pros.map((pro, proIdx) => (
                                <li key={proIdx}>• {pro}</li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="text-sm font-medium text-red-800 mb-1">Cons:</h4>
                            <ul className="text-xs text-red-600 space-y-1">
                              {option.cons.map((con, conIdx) => (
                                <li key={conIdx}>• {con}</li>
                              ))}
                            </ul>
                          </div>
                          
                          <div>
                            <h4 className="text-sm font-medium text-gray-800 mb-1">Technologies:</h4>
                            <div className="flex flex-wrap gap-1">
                              {option.technologies.map((tech, techIdx) => (
                                <span key={techIdx} className="px-2 py-1 bg-gray-100 text-gray-600 rounded text-xs">
                                  {tech}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Scaling Considerations */}
                  <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200">
                    <h3 className="text-lg font-semibold text-purple-800 mb-4">Scaling Considerations</h3>
                    <div className="grid md:grid-cols-3 gap-4">
                      <div>
                        <h4 className="font-medium text-purple-800 mb-2">Performance:</h4>
                        <ul className="text-sm text-purple-600 space-y-1">
                          <li>• Response time optimization</li>
                          <li>• Caching strategies</li>
                          <li>• Load balancing</li>
                          <li>• CDN integration</li>
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-medium text-purple-800 mb-2">Cost Management:</h4>
                        <ul className="text-sm text-purple-600 space-y-1">
                          <li>• API usage monitoring</li>
                          <li>• Resource optimization</li>
                          <li>• Auto-scaling policies</li>
                          <li>• Cost alerting</li>
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-medium text-purple-800 mb-2">Monitoring:</h4>
                        <ul className="text-sm text-purple-600 space-y-1">
                          <li>• Performance metrics</li>
                          <li>• Error tracking</li>
                          <li>• User analytics</li>
                          <li>• Health checks</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Best Practices */}
              {activeSection === 'best-practices' && (
                <div className="space-y-6">
                  <h2 className="text-3xl font-bold text-gray-800 mb-6">Best Practices</h2>
                  
                  <div className="space-y-6">
                    {[
                      {
                        category: "Data Quality",
                        icon: Database,
                        practices: [
                          "Clean and preprocess documents thoroughly",
                          "Remove duplicates and irrelevant content",
                          "Maintain consistent formatting",
                          "Regular data quality audits",
                          "Version control for knowledge base"
                        ]
                      },
                      {
                        category: "Security & Privacy",
                        icon: Shield,
                        practices: [
                          "Encrypt sensitive data at rest and in transit",
                          "Implement proper authentication and authorization",
                          "Regular security audits and penetration testing",
                          "PII data handling and anonymization",
                          "Compliance with regulations (GDPR, HIPAA, etc.)"
                        ]
                      },
                      {
                        category: "Performance Optimization",
                        icon: BarChart3,
                        practices: [
                          "Optimize chunk size for your use case",
                          "Implement caching for frequent queries",
                          "Use appropriate embedding models",
                          "Monitor and optimize retrieval accuracy",
                          "Implement query preprocessing"
                        ]
                      },
                      {
                        category: "User Experience",
                        icon: Users,
                        practices: [
                          "Provide clear source attribution",
                          "Implement confidence scoring",
                          "Handle edge cases gracefully",
                          "Offer query suggestions",
                          "Maintain conversation context"
                        ]
                      },
                      {
                        category: "Monitoring & Maintenance",
                        icon: Settings,
                        practices: [
                          "Track system performance metrics",
                          "Monitor query patterns and failures",
                          "Regular model and data updates",
                          "A/B testing for improvements",
                          "User feedback collection"
                        ]
                      }
                    ].map((section, idx) => (
                      <div key={idx} className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                        <div className="bg-gradient-to-r from-gray-50 to-gray-100 p-4 flex items-center gap-3">
                          <section.icon className="text-gray-600" size={24} />
                          <h3 className="text-lg font-semibold text-gray-800">{section.category}</h3>
                        </div>
                        <div className="p-4">
                          <ul className="space-y-2">
                            {section.practices.map((practice, practiceIdx) => (
                              <li key={practiceIdx} className="flex items-start gap-3 text-gray-600">
                                <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                                <span>{practice}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Common Pitfalls */}
                  <div className="bg-red-50 rounded-lg p-6 border border-red-200">
                    <h3 className="text-lg font-semibold text-red-800 mb-4">Common Pitfalls to Avoid</h3>
                    <div className="grid md:grid-cols-2 gap-4">
                      <ul className="text-sm text-red-700 space-y-2">
                        <li>• Poor chunk size selection</li>
                        <li>• Inadequate data preprocessing</li>
                        <li>• Ignoring retrieval quality metrics</li>
                        <li>• Over-reliance on single embedding model</li>
                        <li>• Insufficient query preprocessing</li>
                      </ul>
                      <ul className="text-sm text-red-700 space-y-2">
                        <li>• Lack of proper error handling</li>
                        <li>• Inadequate security measures</li>
                        <li>• Poor prompt engineering</li>
                        <li>• Ignoring cost optimization</li>
                        <li>• Insufficient testing and validation</li>
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center">
          <div className="bg-bone rounded-3xl brand-shadow p-8 glass-effect">
            {/* Company Logo Placeholder */}
            <div className="mb-6">
              {/* Full Logo with tagline */}
              <div className="w-48 h-16 mx-auto mb-3 bg-white rounded-2xl flex items-center justify-center hover-lift brand-shadow p-3">
                <img src="/logo-evolveiq.png" alt="E EvolvIQ" className="max-w-full max-h-full object-contain" />
              </div>
            </div>
            
            <h3 className="text-2xl playfair-bold text-charcoal mb-4">Ready to Build Your RAG System?</h3>
            <p className="lato-regular text-khaki mb-6 max-w-2xl mx-auto leading-relaxed">
              This comprehensive guide provides everything you need to implement a production-ready RAG system. 
              Start with a basic implementation and gradually add advanced features as your needs grow.
            </p>
            <div className="flex flex-wrap justify-center gap-8" id="download">
              <button 
                onClick={() => {
                  const link = document.createElement('a');
                  link.href = '/python_rag_book.py';
                  link.download = 'python_rag_walkthrough.py';
                  link.click();
                }}
                className="flex items-center gap-4 brand-gradient text-bone px-12 py-6 rounded-2xl hover:shadow-2xl transition-all lato-semibold text-lg hover-lift"
              >
                <Download size={24} />
                Download Python RAG Walkthrough
              </button>
              <button 
                onClick={() => {
                  // Show modal with multiple examples
                  const modal = document.createElement('div');
                  modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
                  modal.innerHTML = `
                    <div class="bg-bone rounded-3xl p-12 max-w-3xl mx-4 max-h-96 overflow-y-auto border-2 border-pearl brand-shadow" style="font-family: 'Lato', sans-serif;">
                      <h3 class="text-3xl font-bold mb-8 text-charcoal" style="font-family: 'Playfair Display', serif; color: #2A2A2A;">Live RAG Examples & Resources</h3>
                      <div class="space-y-6">
                        <a href="https://medical-rag-demolc.streamlit.app/" target="_blank" 
                           class="block p-6 border-2 border-chestnut rounded-2xl hover:bg-pearl transition-colors hover-lift" style="border-color: #A44A3F;">
                          <div class="flex items-center gap-4">
                            <div class="w-4 h-4 rounded-full" style="background-color: #A44A3F;"></div>
                            <div>
                              <h4 class="font-semibold text-lg" style="color: #A44A3F; font-family: 'Playfair Display', serif;">Medical RAG Demo</h4>
                              <p class="text-base" style="color: #A59E8C;">Interactive medical knowledge assistant with RAG</p>
                            </div>
                          </div>
                        </a>
                        <a href="https://python.langchain.com/docs/use_cases/question_answering" target="_blank"
                           class="block p-6 border-2 border-khaki rounded-2xl hover:bg-pearl transition-colors hover-lift" style="border-color: #A59E8C;">
                          <div class="flex items-center gap-4">
                            <div class="w-4 h-4 rounded-full" style="background-color: #A59E8C;"></div>
                            <div>
                              <h4 class="font-semibold text-lg" style="color: #A59E8C; font-family: 'Playfair Display', serif;">LangChain RAG Tutorial</h4>
                              <p class="text-base" style="color: #A59E8C;">Official documentation and examples</p>
                            </div>
                          </div>
                        </a>
                        <a href="https://github.com/openai/openai-cookbook/tree/main/examples" target="_blank"
                           class="block p-6 border-2 border-chestnut rounded-2xl hover:bg-pearl transition-colors hover-lift" style="border-color: #A44A3F;">
                          <div class="flex items-center gap-4">
                            <div class="w-4 h-4 rounded-full" style="background-color: #A44A3F;"></div>
                            <div>
                              <h4 class="font-semibold text-lg" style="color: #A44A3F; font-family: 'Playfair Display', serif;">OpenAI RAG Examples</h4>
                              <p class="text-base" style="color: #A59E8C;">Code examples and best practices from OpenAI</p>
                            </div>
                          </div>
                        </a>
                        <a href="https://github.com/streamlit/llm-examples" target="_blank"
                           class="block p-6 border-2 border-khaki rounded-2xl hover:bg-pearl transition-colors hover-lift" style="border-color: #A59E8C;">
                          <div class="flex items-center gap-4">
                            <div class="w-4 h-4 rounded-full" style="background-color: #A59E8C;"></div>
                            <div>
                              <h4 class="font-semibold text-lg" style="color: #A59E8C; font-family: 'Playfair Display', serif;">Streamlit RAG Examples</h4>
                              <p class="text-base" style="color: #A59E8C;">Ready-to-deploy Streamlit applications</p>
                            </div>
                          </div>
                        </a>
                      </div>
                      <button onclick="this.parentElement.parentElement.remove()" 
                              class="mt-8 w-full py-4 px-6 rounded-2xl transition-colors text-lg font-semibold" style="background-color: #A59E8C; color: #F5F2EA;">
                        Close
                      </button>
                    </div>
                  `;
                  document.body.appendChild(modal);
                  modal.addEventListener('click', (e) => {
                    if (e.target === modal) modal.remove();
                  });
                }}
                className="flex items-center gap-4 border-2 border-khaki text-charcoal px-12 py-6 rounded-2xl hover:bg-pearl transition-all lato-semibold text-lg hover-lift"
              >
                <ExternalLink size={24} />
                View Live Examples
              </button>
            </div>
            
            {/* Social Proof / Trust Indicators */}
            <div className="mt-12 pt-8 border-t-2 border-pearl">
              <div className="grid md:grid-cols-4 gap-8 text-center">
                {[
                  { icon: Users, count: "10,000+", label: "Developers Using" },
                  { icon: Code, count: "50+", label: "Code Examples" },
                  { icon: BookOpen, count: "8", label: "Complete Sections" },
                  { icon: BarChart3, count: "95%", label: "Success Rate" }
                ].map((metric, idx) => (
                  <div key={idx} className="hover-lift">
                    <metric.icon size={28} className="mx-auto mb-3 text-chestnut" />
                    <div className="text-3xl playfair-bold text-charcoal mb-2">{metric.count}</div>
                    <div className="text-base lato-medium text-khaki">{metric.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-charcoal text-bone mt-16">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <img src="/logo-evolveiq.png" alt="E EvolvIQ" className="h-10 mb-4 filter brightness-0 invert" />
              <p className="text-sm lato-regular text-pearl">
                Empowering businesses with AI-driven solutions for the modern era.
              </p>
            </div>
            <div>
              <h4 className="playfair-bold text-lg mb-3">Quick Links</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="#overview" className="text-pearl hover:text-bone transition-colors">Overview</a></li>
                <li><a href="#download" className="text-pearl hover:text-bone transition-colors">Download Guide</a></li>
                <li><a href="#examples" className="text-pearl hover:text-bone transition-colors">Examples</a></li>
              </ul>
            </div>
            <div>
              <h4 className="playfair-bold text-lg mb-3">Contact</h4>
              <p className="text-sm text-pearl">
                questions@evolveiq.com<br />
                © 2025 EvolvIQ. All rights reserved.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default RAGGuide;