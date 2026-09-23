const ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "inc", "ltd", "corp",
  "co", "e.g", "i.e", "vs", "etc", "approx", "dept", "vol", "no",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
]);

class FastJarvisSentenceTokenizer {
  constructor() {
    this.buffer = "";
    this.isFirstSentence = true;
  }

  feed(chunk) {
    this.buffer += chunk;
    const sentences = [];

    while (this.buffer.length > 0) {
      const match = this.findSentenceEnd(this.buffer);
      if (match === -1) {
        break;
      }

      const sentence = this.buffer.slice(0, match + 1).trim();
      this.buffer = this.buffer.slice(match + 1);

      if (sentence.length > 0) {
        sentences.push(sentence);
        this.isFirstSentence = false;
      }
    }

    return sentences;
  }

  flush() {
    const remaining = this.buffer.trim();
    this.buffer = "";
    this.isFirstSentence = true;
    return remaining.length > 0 ? remaining : null;
  }

  findSentenceEnd(text) {
    for (let i = 0; i < text.length; i++) {
      const char = text[i];

      // Double newline or newline before bullet points / list items
      if (char === "\n") {
        if (i + 1 < text.length && (text[i + 1] === "\n" || text[i + 1] === "-" || text[i + 1] === "*")) {
          return i;
        }
        if (i >= 12) {
          return i;
        }
      }

      // Colon followed by whitespace or end
      if (char === ":" && i >= 8) {
        if (i + 1 === text.length || /\s/.test(text[i + 1])) {
          return i;
        }
      }

      // Semicolon followed by whitespace
      if (char === ";" && i >= 10) {
        if (i + 1 === text.length || /\s/.test(text[i + 1])) {
          return i;
        }
      }

      // Devanagari Danda (।) or Double Danda (॥)
      if (char === "\u0964" || char === "\u0965") {
        return i;
      }

      // Question mark or exclamation mark
      if (char === "?" || char === "!") {
        return i;
      }

      // Em-dash or en-dash
      if ((char === "—" || char === "–") && i >= 10) {
        return i;
      }

      // Fast-path Jarvis-like first clause break:
      // If this is the very first sentence chunk:
      // Allow comma after >= 12 characters, or natural word boundary (space) after >= 28 characters
      if (this.isFirstSentence) {
        if (char === "," && i >= 12) {
          const isNumberComma = i > 0 && /\d/.test(text[i - 1]) && i + 1 < text.length && /\d/.test(text[i + 1]);
          if (!isNumberComma && (i + 1 === text.length || /\s/.test(text[i + 1]))) {
            return i;
          }
        }
        if (char === " " && i >= 28) {
          return i;
        }
      }

      // Period (.)
      if (char === ".") {
        if (i + 1 < text.length && text[i + 1] === ".") continue;
        if (i > 0 && text[i - 1] === ".") continue;

        if (i > 0 && /\d/.test(text[i - 1])) {
          if (i + 1 < text.length && /\d/.test(text[i + 1])) continue;
          if (i + 1 === text.length) continue;
        }

        const precedingWordMatch = text.slice(0, i).match(/([a-zA-Z]+)$/);
        if (precedingWordMatch) {
          const word = precedingWordMatch[1].toLowerCase();
          if (ABBREVIATIONS.has(word) || word.length === 1) continue;
        }

        if (i + 1 === text.length || /\s/.test(text[i + 1])) {
          if (i >= 4) return i;
        }
      }
    }

    return -1;
  }
}

const testInputs = [
  "For personal loans, the minimum CIBIL score required is 750 or above. Borrowers with lower scores may require additional collateral.",
  "The minimum CIBIL score required for a personal loan is typically 750 or above, though some lenders accept 650. You should check eligibility.",
  "एचडीएफसी बँकेच्या नियमांनुसार, वैयक्तिक कर्जासाठी किमान सिबिल स्कोअर 750 आवश्यक आहे. कागदपत्रे वेळेवर सादर करा.",
  "Under our enterprise volume policy: Tier 1 gives 15% discount for 500+ users. Tier 2 gives 25% discount for 1000+ users."
];

for (const input of testInputs) {
  const tokenizer = new FastJarvisSentenceTokenizer();
  const chunks = [];
  // simulate streaming word by word
  const words = input.split(" ");
  for (const w of words) {
    const s = tokenizer.feed(w + " ");
    chunks.push(...s);
  }
  const trailing = tokenizer.flush();
  if (trailing) chunks.push(trailing);

  console.log("\nINPUT:", input);
  console.log("CHUNKS (" + chunks.length + "):");
  chunks.forEach((c, idx) => console.log(`  [Chunk #${idx} (${c.length} chars)]: "${c}"`));
}
