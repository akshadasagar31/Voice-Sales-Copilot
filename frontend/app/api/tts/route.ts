// ============================================================================
// NEXT.JS API GATEWAY: TEXT-TO-SPEECH (/api/tts)
// ============================================================================
// This API route accepts a sentence of text and target language ('en', 'hi', 'mr'),
// sends it to the FastAPI TTS service, and returns raw MP3 audio bytes.
//
// DATA FLOW:
// 1. SentenceAudioQueue in frontend feeds a single completed sentence to this route.
// 2. This route forwards the text to FastAPI: POST /api/tts.
// 3. FastAPI calls Deepgram Aura TTS with selected voice model.
// 4. Returns MP3 binary audio (`audio/mpeg`) to browser, which creates a Blob URL
//    and plays it immediately through HTMLAudioElement.
// ============================================================================

import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const text = (body?.text || "").trim();

    // Validate that text is not empty
    if (!text) {
      return NextResponse.json(
        { error: "The 'text' field cannot be empty or whitespace." },
        { status: 400 }
      );
    }

    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/tts`;


    const backendRes = await fetch(targetUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        model: body?.model || undefined,
        language: body?.language || null,
        module: body?.module || undefined,
        speaker: body?.speaker || undefined,
      }),
    });

    if (!backendRes.ok) {
      const errText = await backendRes.text();
      let errDetail = errText;
      try {
        const errJson = JSON.parse(errText);
        errDetail = errJson.detail || errJson.error || errDetail;
      } catch {
        // use raw text
      }

      return NextResponse.json(
        {
          error: errDetail || "FastAPI TTS synthesis returned an error.",
          status: backendRes.status,
        },
        { status: backendRes.status }
      );
    }

    const audioBuffer = await backendRes.arrayBuffer();
    const contentType = backendRes.headers.get("content-type") || "audio/mpeg";
    const ext = contentType.includes("wav") ? "wav" : "mp3";

    return new NextResponse(audioBuffer, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Content-Disposition": `inline; filename=tts_response.${ext}`,
      },
    });
  } catch (error: any) {
    console.error("Error proxying request to FastAPI /api/tts:", error);
    return NextResponse.json(
      {
        error:
          "Unable to connect to FastAPI TTS service. Please ensure the backend is running at http://127.0.0.1:8001.",
        details: error.message,
      },
      { status: 503 }
    );
  }
}
