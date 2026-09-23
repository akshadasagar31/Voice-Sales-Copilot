// ============================================================================
// NEXT.JS EDGE MIDDLEWARE (frontend/middleware.ts)
// ============================================================================
// Next.js middleware runs BEFORE every matched page request reaches the server.
// What it does:
// 1. Checks if the incoming request has a valid 'auth_token' cookie.
// 2. If an unauthenticated user tries to visit protected pages (/dashboard/*),
//    it redirects them to /login with a redirect query parameter so they return after login.
// 3. If an already-logged-in user visits /login or /register, it forwards them
//    straight to /dashboard so they don't see login forms again.
// ============================================================================

import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { jwtVerify } from "jose";

// Cookie name where the session token is stored
const AUTH_COOKIE_NAME = "auth_token";
const JWT_SECRET_STRING =
  process.env.JWT_SECRET || "voice_sales_copilot_jwt_super_secret_key_2026_default_dev";
const JWT_KEY = new TextEncoder().encode(JWT_SECRET_STRING);

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  // Read the auth_token cookie from browser request headers
  const token = request.cookies.get(AUTH_COOKIE_NAME)?.value;

  // Verify the JWT cryptographic signature
  let isValidSession = false;
  if (token) {
    try {
      await jwtVerify(token, JWT_KEY);
      isValidSession = true;
    } catch {
      // Token is invalid, tampered with, or expired
      isValidSession = false;
    }
  }

  // 1. Route Protection: Redirect unauthenticated users away from /dashboard
  if (pathname.startsWith("/dashboard")) {
    if (!isValidSession) {
      const loginUrl = new URL("/login", request.url);
      loginUrl.searchParams.set("redirect", pathname); // Save where they wanted to go
      return NextResponse.redirect(loginUrl);
    }
  }

  // 2. Convenience: If already logged in, skip the register page
  if (isValidSession && pathname === "/register") {
    return NextResponse.redirect(new URL("/dashboard", request.url));
  }

  // Allow the request to proceed normally
  return NextResponse.next();
}

// Specify which URL routes this middleware runs on
export const config = {
  matcher: ["/dashboard/:path*", "/register"],
};

