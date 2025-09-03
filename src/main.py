#!/usr/bin/env python3
"""
Enterprise AI Analytics Platform (EAAP) - Main Application Entry Point

This is the main entry point for the EAAP system, implementing the Three-Layer
AI Architecture for enterprise data analytics and insights generation.

Author: Ram Senthil-Maree
License: MIT
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from api.routes import api_router
from api.middleware import setup_middleware
from api.dependencies import get_current_user
from ml_pipeline.automl_engine import AutoMLEngine
from data_pipeline.ingestion_engine import DataIngestionEngine
from llm_interface.query_engine import NaturalLanguageQueryEngine
from dashboard.dashboard_engine import DynamicDashboardEngine
from deployment.config import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global settings
settings = Settings()

# Global engine instances
ml_engine: AutoMLEngine = None
data_engine: DataIngestionEngine = None
query_engine: NaturalLanguageQueryEngine = None
dashboard_engine: DynamicDashboardEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown events
    """
    # Startup
    logger.info("Starting Enterprise AI Analytics Platform...")
    
    global ml_engine, data_engine, query_engine, dashboard_engine
    
    try:
        # Initialize engines
        logger.info("Initializing AI engines...")
        ml_engine = AutoMLEngine(settings.ml_config)
        data_engine = DataIngestionEngine(settings.data_config)
        query_engine = NaturalLanguageQueryEngine(settings.llm_config)
        dashboard_engine = DynamicDashboardEngine(settings.dashboard_config)
        
        # Health checks
        await ml_engine.health_check()
        await data_engine.health_check()
        await query_engine.health_check()
        await dashboard_engine.health_check()
        
        logger.info("✅ All engines initialized successfully")
        
        # Start background tasks
        await data_engine.start_ingestion_pipeline()
        await ml_engine.start_model_monitoring()
        
        logger.info("🚀 EAAP is ready to serve requests!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize EAAP: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enterprise AI Analytics Platform...")
    
    try:
        # Stop background tasks
        await data_engine.stop_ingestion_pipeline()
        await ml_engine.stop_model_monitoring()
        
        # Cleanup resources
        await ml_engine.cleanup()
        await data_engine.cleanup()
        await query_engine.cleanup()
        await dashboard_engine.cleanup()
        
        logger.info("✅ EAAP shut down gracefully")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="Enterprise AI Analytics Platform",
    description="Production-ready AI analytics with Three-Layer Architecture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint - serves the main landing page
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enterprise AI Analytics Platform</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
            .header { text-align: center; color: #2563eb; margin-bottom: 30px; }
            .status { padding: 20px; background: #10b981; color: white; border-radius: 5px; text-align: center; }
            .features { margin-top: 30px; }
            .feature { margin: 20px 0; padding: 15px; background: #f8fafc; border-radius: 5px; }
            .links { margin-top: 30px; text-align: center; }
            .links a { margin: 0 10px; padding: 10px 20px; background: #2563eb; color: white; text-decoration: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 Enterprise AI Analytics Platform</h1>
                <p>Production-ready AI analytics with Three-Layer Architecture</p>
            </div>
            
            <div class="status">
                ✅ System is online and ready to serve requests
            </div>
            
            <div class="features">
                <h2>Available Features:</h2>
                <div class="feature">
                    <h3>🔍 Natural Language Queries</h3>
                    <p>Ask questions in plain English and get intelligent insights</p>
                </div>
                <div class="feature">
                    <h3>🤖 AutoML Pipeline</h3>
                    <p>Automated machine learning model training and deployment</p>
                </div>
                <div class="feature">
                    <h3>📊 Dynamic Dashboards</h3>
                    <p>AI-generated visualizations based on your data</p>
                </div>
                <div class="feature">
                    <h3>⚡ Real-time Processing</h3>
                    <p>Stream processing and live data ingestion</p>
                </div>
            </div>
            
            <div class="links">
                <a href="/docs">📚 API Documentation</a>
                <a href="/api/v1/health">🔍 Health Check</a>
                <a href="/api/v1/demo">🎯 Live Demo</a>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers
    """
    try:
        # Check all engine health
        ml_status = await ml_engine.health_check() if ml_engine else False
        data_status = await data_engine.health_check() if data_engine else False
        query_status = await query_engine.health_check() if query_engine else False
        dashboard_status = await dashboard_engine.health_check() if dashboard_engine else False
        
        health_data = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",  # Will be updated with real timestamp
            "version": "1.0.0",
            "services": {
                "ml_engine": "healthy" if ml_status else "unhealthy",
                "data_engine": "healthy" if data_status else "unhealthy", 
                "query_engine": "healthy" if query_status else "unhealthy",
                "dashboard_engine": "healthy" if dashboard_status else "unhealthy"
            },
            "system": {
                "cpu_usage": "< 50%",
                "memory_usage": "< 70%",
                "disk_usage": "< 80%"
            }
        }
        
        return JSONResponse(content=health_data)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")


@app.get("/api/v1/demo")
async def demo_endpoint():
    """
    Demo endpoint showcasing the Three-Layer Architecture
    """
    return {
        "message": "Enterprise AI Analytics Platform Demo",
        "architecture": {
            "layer_1": {
                "name": "User Experience & Interaction",
                "features": ["Natural Language Queries", "Dynamic Dashboards", "Automated Reports"]
            },
            "layer_2": {
                "name": "Data & Knowledge Intelligence", 
                "features": ["Real-time Ingestion", "Knowledge Graphs", "Feature Engineering"]
            },
            "layer_3": {
                "name": "Strategic Intelligence & ML",
                "features": ["AutoML Pipeline", "Ensemble Predictions", "Scenario Planning"]
            }
        },
        "sample_queries": [
            "Show me revenue trends for the last 6 months",
            "Which customers are at risk of churning?", 
            "Generate a sales performance report",
            "What are the key factors driving customer satisfaction?"
        ]
    }


async def main():
    """
    Main function for running the application
    """
    logger.info("Starting Enterprise AI Analytics Platform...")
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("ENVIRONMENT", "production") == "development"
    
    # Start server
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())