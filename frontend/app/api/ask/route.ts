// ============================================================================
// NEXT.JS API GATEWAY: RAG QUESTION ANSWERING (/api/ask)
// ============================================================================
// This API route acts as a bridge between the frontend React application and
// the Python FastAPI backend service running at port 8001.
//
// DATA FLOW:
// 1. Browser sends user query (question, language, streaming preference).
// 2. This route validates the query text and forwards it to FastAPI: POST /api/ask.
// 3. If streaming is requested (`stream: true`), this route transparently pipes
//    the Server-Sent Events (SSE) byte stream directly to the browser.
// 4. If non-streaming, it waits for the full JSON response and returns it.
// ============================================================================

import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { question, top_k, namespace, filter, model, language, stream } = body;

    // Validate that the question is non-empty
    if (!question || typeof question !== "string" || !question.trim()) {
      return NextResponse.json(
        { error: "The 'question' parameter is required and cannot be empty." },
        { status: 400 }
      );
    }

    // Determine the FastAPI backend URL (defaults to localhost:8001)
    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/ask`;

    // Forward the request to FastAPI backend
    const backendRes = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question.trim(),
        top_k: top_k || 8, // Retrieve top 8 most similar document chunks from Pinecone
        namespace: namespace && namespace.trim() ? namespace.trim() : "sales_playbooks",
        filter: filter || null,
        model: model || null,
        language: language || null,
        stream: Boolean(stream),
      }),
    });

    if (!backendRes.ok) {
      const data = await backendRes.json().catch(() => ({}));
      return NextResponse.json(
        {
          error: data.detail || data.error || "RAG backend returned an error.",
          status: backendRes.status,
        },
        { status: backendRes.status }
      );
    }


    // If streaming mode was requested, pipe the SSE stream directly to client
    if (Boolean(stream) && backendRes.body) {
      return new Response(backendRes.body, {
        headers: {
          "Content-Type": "text/event-stream; charset=utf-8",
          "Cache-Control": "no-cache, no-transform",
          "Connection": "keep-alive",
          "X-Accel-Buffering": "no",
        },
      });
    }

    const data = await backendRes.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error: any) {
    console.error("Error proxying request to FastAPI /api/ask:", error);
    return NextResponse.json(
      {
        error:
          "Unable to connect to FastAPI RAG service. Please ensure the backend is running at http://127.0.0.1:8001.",
        details: error.message,
      },
      { status: 503 }
    );
  }
}
