// ============================================================================
// NEXT.JS API GATEWAY: LEAD EXTRACTION (/api/extract-lead)
// ============================================================================
// This API route receives audio transcripts from the frontend, forwards them
// to the FastAPI lead extraction service, and returns structured sales lead data.
//
// DATA FLOW:
// 1. User speaks in Module 1 -> Deepgram STT transcribes speech to text.
// 2. Browser calls POST /api/extract-lead with:
//    - transcript: speech text from caller
//    - existing_lead: current lead data in state (to avoid creating duplicates!)
//    - stream: true (for fast word-by-word streaming voice confirmation)
//    - lead_id: database ID if already created in PostgreSQL
// 3. This route pipes the SSE events (lead updates + token stream) to the browser.
// ============================================================================

import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const transcript = body?.transcript;

    // Validate that transcript text is provided
    if (!transcript || typeof transcript !== "string" || !transcript.trim()) {
      return NextResponse.json(
        { error: "Transcript is required and cannot be empty." },
        { status: 400 }
      );
    }

    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/extract-lead`;

    // Forward the payload to FastAPI backend
    const backendRes = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });

    if (!backendRes.ok) {
      const data = await backendRes.json().catch(() => ({}));
      return NextResponse.json(
        {
          error: data.detail?.message || data.detail || data.error || "FastAPI lead extraction failed.",
          status: backendRes.status,
        },
        { status: backendRes.status }
      );
    }


    // If streaming mode was requested, pipe the SSE stream directly to client
    if (Boolean(body.stream) && backendRes.body) {
      const responseStream = new ReadableStream({
        async start(controller) {
          const reader = backendRes.body!.getReader();
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              controller.enqueue(value);
            }
          } catch (err) {
            controller.error(err);
          } finally {
            controller.close();
          }
        },
      });

      return new Response(responseStream, {
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
    console.error("Error proxying request to FastAPI /api/extract-lead:", error);
    return NextResponse.json(
      {
        error:
          "Unable to connect to FastAPI lead extraction service. Please ensure the backend is running at http://127.0.0.1:8001.",
        details: error.message,
      },
      { status: 503 }
    );
  }
}
