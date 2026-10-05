import assert from "node:assert/strict";
import test from "node:test";

// Node's type-strip test runner requires the source extension at runtime.
// @ts-ignore TS5097 is not applicable to the production bundler import graph.
import {
  getSarvamSpeaker,
  getSpeechLanguageCode,
  normalizeSpeechLanguage,
} from "./sentenceStreamingTTS.ts";

test("normalizes Hindi and Marathi language variants for Module 1 playback", () => {
  assert.equal(normalizeSpeechLanguage("hi-IN"), "hi");
  assert.equal(normalizeSpeechLanguage("mr_IN"), "mr");
  assert.equal(getSpeechLanguageCode("hindi"), "hi-IN");
  assert.equal(getSpeechLanguageCode("marathi"), "mr-IN");
});

test("selects only the matching Sarvam Indic voice", () => {
  assert.equal(getSarvamSpeaker("hi-IN"), "priya");
  assert.equal(getSarvamSpeaker("mr-IN"), "ritu");
  assert.notEqual(getSarvamSpeaker("hi-IN"), "simran");
  assert.notEqual(getSarvamSpeaker("mr-IN"), "simran");
});
