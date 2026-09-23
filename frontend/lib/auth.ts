// ============================================================================
// AUTHENTICATION UTILITIES (frontend/lib/auth.ts)
// ============================================================================
// This file handles user password hashing and JWT (JSON Web Token) creation.
// Why we use this:
// 1. Passwords should NEVER be saved in plain text in the database. We use bcrypt
//    to securely hash them with salt before saving.
// 2. When a user logs in, we issue a signed JWT token stored in an HTTP-only cookie.
//    Every time the user visits a page or makes an API request, Next.js verifies
//    this token to confirm who is logged in without querying the database every time.
// ============================================================================

import bcrypt from "bcryptjs";
import { SignJWT, jwtVerify } from "jose";

// Name of the cookie where the user session token is stored in the browser
export const AUTH_COOKIE_NAME = "auth_token";

// Secret key used to cryptographically sign and verify JWT tokens.
// In production, this must be set in .env. For local development, a fallback is provided.
const JWT_SECRET_STRING =
  process.env.JWT_SECRET || "voice_sales_copilot_jwt_super_secret_key_2026_default_dev";
const JWT_KEY = new TextEncoder().encode(JWT_SECRET_STRING);

// The user data stored inside the signed JWT token payload
export interface UserSessionPayload {
  id: number;
  name: string;
  email: string;
  company?: string;
  role?: string;
}

/**
 * Hashes a plaintext password using bcrypt (cost factor 10).
 * Salt is automatically generated to defend against rainbow table attacks.
 */
export async function hashPassword(password: string): Promise<string> {
  return bcrypt.hash(password, 10);
}

/**
 * Compares a plaintext password entered by the user with the stored bcrypt hash.
 * Returns true if the password matches, or false otherwise.
 */
export async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

/**
 * Creates and cryptographically signs a JWT session token valid for 7 days.
 * The payload contains user identification details (id, name, email).
 */
export async function signJwtToken(payload: UserSessionPayload): Promise<string> {
  return new SignJWT({ ...payload })
    .setProtectedHeader({ alg: "HS256" }) // HMAC SHA-256 digital signature
    .setIssuedAt()
    .setExpirationTime("7d")              // Token expires after 7 days
    .sign(JWT_KEY);
}

/**
 * Verifies a JWT session token received from the user's browser cookie.
 * If the signature is valid and not expired, returns the decoded user data.
 * If invalid or expired, gracefully returns null (indicating unauthenticated).
 */
export async function verifyJwtToken(token: string): Promise<UserSessionPayload | null> {
  try {
    const { payload } = await jwtVerify(token, JWT_KEY);
    return payload as unknown as UserSessionPayload;
  } catch {
    // If token is expired, tampered with, or invalid, treat user as logged out
    return null;
  }
}

