import { NextRequest, NextResponse } from "next/server";
import { verifyJwtToken, AUTH_COOKIE_NAME } from "@/lib/auth";

export async function GET(request: NextRequest) {
  const token = request.cookies.get(AUTH_COOKIE_NAME)?.value;

  if (!token) {
    return NextResponse.json(
      { authenticated: false, error: "No session token provided" },
      { status: 401 }
    );
  }

  const payload = await verifyJwtToken(token);

  if (!payload) {
    return NextResponse.json(
      { authenticated: false, error: "Invalid or expired session token" },
      { status: 401 }
    );
  }

  return NextResponse.json({
    authenticated: true,
    user: payload,
  });
}
