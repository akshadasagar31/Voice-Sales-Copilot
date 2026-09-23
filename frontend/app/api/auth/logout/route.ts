import { NextResponse } from "next/server";
import { AUTH_COOKIE_NAME } from "@/lib/auth";

function clearSessionCookie(response: NextResponse) {
  response.cookies.set({
    name: AUTH_COOKIE_NAME,
    value: "",
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "lax",
    path: "/",
    maxAge: 0,
    expires: new Date(0),
  });
}

export async function POST() {
  const response = NextResponse.json({
    success: true,
    message: "Logged out successfully",
  });
  clearSessionCookie(response);
  return response;
}

export async function GET() {
  const response = NextResponse.json({
    success: true,
    message: "Logged out successfully",
  });
  clearSessionCookie(response);
  return response;
}

